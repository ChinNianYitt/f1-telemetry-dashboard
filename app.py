import os
import tempfile
import streamlit as st
import fastf1
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import plotly.subplots as sp
from fastf1 import utils
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from datetime import datetime, timezone
import requests
from fastf1.core import Laps, Telemetry

# --- FASTF1 CACHE SETUP ---
cache_dir = os.path.join(tempfile.gettempdir(), "f1_cache")
os.makedirs(cache_dir, exist_ok=True)
try:
    fastf1.Cache.enable_cache(cache_dir)
except Exception:
    pass

# --- STREAMLIT PAGE CONFIG ---
st.set_page_config(page_title="F1 Telemetry Dashboard", layout="wide")

st.title("🏎️ Formula 1 Race & Telemetry Dashboard")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Session Settings")

current_year = datetime.now().year

year = st.sidebar.number_input(
    "Year", 
    min_value=2018, 
    max_value=current_year, 
    value=current_year, 
    step=1
)

st.sidebar.caption("⚠️ *Full telemetry and GPS data are only available from 2018 onward.*")

@st.cache_data
def get_standings_after_race(year, round_number):
    try:
        wdc_url = f"https://api.jolpi.ca/ergast/f1/{year}/{round_number}/driverStandings.json"
        wcc_url = f"https://api.jolpi.ca/ergast/f1/{year}/{round_number}/constructorStandings.json"
        
        wdc_res = requests.get(wdc_url, timeout=5).json()
        wcc_res = requests.get(wcc_url, timeout=5).json()
        
        # 1. Parse WDC
        driver_standings_list = wdc_res['MRData']['StandingsTable']['StandingsLists'][0]['DriverStandings']
        wdc_data = []
        for item in driver_standings_list:
            wdc_data.append({
                "Pos": int(item['position']),
                "Driver": f"{item['Driver']['givenName']} {item['Driver']['familyName']}",
                "Code": item['Driver'].get('code', item['Driver']['familyName'][:3].upper()),
                "Team": item['Constructors'][0]['name'],
                "Points": float(item['points']),
                "Wins": int(item['wins'])
            })
        df_wdc = pd.DataFrame(wdc_data)

        # 2. Parse WCC
        constructor_standings_list = wcc_res['MRData']['StandingsTable']['StandingsLists'][0]['ConstructorStandings']
        wcc_data = []
        for item in constructor_standings_list:
            wcc_data.append({
                "Pos": int(item['position']),
                "Team": item['Constructor']['name'],
                "Points": float(item['points']),
                "Wins": int(item['wins'])
            })
        df_wcc = pd.DataFrame(wcc_data)

        return df_wdc, df_wcc
    except Exception:
        return None, None


# --- DYNAMIC GRAND PRIX CALENDAR ---
@st.cache_data
def get_season_calendar_and_default(selected_year):
    try:
        schedule = fastf1.get_event_schedule(selected_year)
        gp_events = schedule[schedule['RoundNumber'] > 0].copy()
        event_names = gp_events['EventName'].tolist()
        
        now = pd.Timestamp.now(tz='UTC')
        event_dates = pd.to_datetime(gp_events['EventDate'], utc=True)
        past_events = gp_events[event_dates <= now]

        if not past_events.empty:
            latest_past_event_name = past_events.iloc[-1]['EventName']
            default_index = event_names.index(latest_past_event_name) if latest_past_event_name in event_names else 0
        else:
            default_index = 0

        return event_names, default_index
    except Exception:
        return ["Australian Grand Prix", "Azerbaijan Grand Prix", "Bahrain Grand Prix"], 0

available_gps, default_gp_idx = get_season_calendar_and_default(year)

grand_prix = st.sidebar.selectbox(
    "Grand Prix", 
    available_gps, 
    index=default_gp_idx
)

session_type = st.sidebar.selectbox("Session", ["R", "Q"], format_func=lambda x: "Race" if x == "R" else "Qualifying")

SHOWCASE_DIR = "showcase_data"

class MockSession:
    def __init__(self, laps_df, results_df, telemetry_df=None, round_num=1):
        self.laps = Laps(laps_df)
        self.results = results_df
        self.telemetry = telemetry_df
        self.drivers = list(results_df['Abbreviation']) if 'Abbreviation' in results_df else []
        self.event = {'RoundNumber': round_num}

    def get_driver_telemetry(self, driver_abbr):
        if self.telemetry is not None and not self.telemetry.empty:
            driver_data = self.telemetry[self.telemetry['Driver'] == str(driver_abbr)]
            if not driver_data.empty:
                return Telemetry(driver_data)
        return None

@st.cache_data(show_spinner=False)
def load_session(year, grand_prix, session_code):
    gp_query = str(grand_prix).lower().replace("grand prix", "").strip().replace(" ", "_")
    
    # 1. Check offline Parquet directory
    if os.path.exists(SHOWCASE_DIR):
        for f in os.listdir(SHOWCASE_DIR):
            if f.startswith(f"{year}_") and (gp_query in f.lower()) and f.endswith("_laps.parquet"):
                base_slug = f.replace("_laps.parquet", "")
                try:
                    laps_df = pd.read_parquet(os.path.join(SHOWCASE_DIR, f"{base_slug}_laps.parquet"))
                    results_df = pd.read_parquet(os.path.join(SHOWCASE_DIR, f"{base_slug}_results.parquet"))
                    tel_path = os.path.join(SHOWCASE_DIR, f"{base_slug}_telemetry.parquet")
                    tel_df = pd.read_parquet(tel_path) if os.path.exists(tel_path) else None
                    
                    # Map official 2026 round numbers
                    round_num = 1
                    if "azerbaijan" in base_slug or "baku" in base_slug:
                        round_num = 15
                    elif "bahrain" in base_slug or "sepang" in base_slug:
                        round_num = 16
                    else:
                        parts = base_slug.split("_")
                        if len(parts) > 1 and parts[1].isdigit():
                            round_num = int(parts[1])
                        
                    return MockSession(laps_df, results_df, tel_df, round_num), None
                except Exception:
                    pass

    # 2. Online fallback
    try:
        session = fastf1.get_session(year, grand_prix, session_code)
        session.load(telemetry=True, laps=True, weather=False)
        return session, None
    except Exception as e:
        return None, str(e)

with st.spinner("Fetching F1 session data..."):
    session, load_error = load_session(year, grand_prix, session_type)

has_laps = False
if session is not None and load_error is None:
    try:
        if not session.laps.empty:
            has_laps = True
    except Exception:
        has_laps = False

if not has_laps:
    st.error(f"⚠️ No session data available for the **{year} {grand_prix}**.")
    st.info(
        "This race either hasn't taken place yet or official timing data has not been published by Formula 1. "
        "Please select a past Grand Prix from the sidebar."
    )
    st.stop()

drivers = sorted(session.laps['Driver'].dropna().unique().tolist())
selected_driver = st.sidebar.selectbox("Select Primary Driver", drivers, index=0)

# --- TAB LAYOUT ---
tab_track, tab_stints, tab_telemetry, tab_positions, tab_deg, tab_standings, tab_teammates = st.tabs([
    "🗺️ Track Map", "🛞 Stints & Tires", "📈 Speed Telemetry", "🏁 Race Progression", 
    "🤖 Tire Degradation (ML)", "🏆 Standings", "👥 Teammate Battles"
])

# 1. TRACK MAP & DOMINANCE HEATMAP
with tab_track:
    st.subheader(f"Track Analysis - {grand_prix} ({year})")
    
    map_mode = st.radio(
        "Display Mode",
        ["Speed Heatmap", "Head-to-Head Track Dominance"],
        horizontal=True
    )

    if map_mode == "Speed Heatmap":
        map_driver = st.selectbox("Select Driver for Speed Heatmap", drivers, index=0)
        driver_laps = session.laps.pick_driver(map_driver)
        
        if not driver_laps.empty:
            fastest_lap = driver_laps.pick_fastest()
            tel = None
            if hasattr(session, "get_driver_telemetry") and session.get_driver_telemetry(map_driver) is not None:
                tel = session.get_driver_telemetry(map_driver)
            elif fastest_lap is not None:
                try:
                    tel = fastest_lap.get_telemetry()
                except Exception:
                    pass

            if tel is not None and "X" in tel.columns and "Y" in tel.columns:
                fig_heatmap = px.scatter(
                    tel,
                    x="X",
                    y="Y",
                    color="Speed",
                    color_continuous_scale="Turbo",
                    title=f"Speed Heatmap: {map_driver} ({fastest_lap['LapTime'] if fastest_lap is not None else ''})",
                    labels={"Speed": "km/h"}
                )
                fig_heatmap.update_yaxes(scaleanchor="x", scaleratio=1)
                fig_heatmap.update_traces(marker=dict(size=4))
                fig_heatmap.update_layout(
                    template="plotly_dark",
                    xaxis=dict(showgrid=False, zeroline=False, visible=False),
                    yaxis=dict(showgrid=False, zeroline=False, visible=False),
                    height=700
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                st.warning(f"GPS Track map coordinates (X, Y) are unavailable for {map_driver}.")
        else:
            st.warning(f"No lap data available for {map_driver}.")

    else:
        col_m1, col_m2 = st.columns(2)
        dom_d1 = col_m1.selectbox("Driver 1 (Dominance)", drivers, index=0)
        dom_d2 = col_m2.selectbox("Driver 2 (Dominance)", drivers, index=1 if len(drivers) > 1 else 0)

        tel1 = session.get_driver_telemetry(dom_d1) if hasattr(session, "get_driver_telemetry") else None
        tel2 = session.get_driver_telemetry(dom_d2) if hasattr(session, "get_driver_telemetry") else None

        if tel1 is None:
            l1 = session.laps.pick_driver(dom_d1).pick_fastest()
            if l1 is not None:
                try: tel1 = l1.get_telemetry()
                except Exception: pass
        if tel2 is None:
            l2 = session.laps.pick_driver(dom_d2).pick_fastest()
            if l2 is not None:
                try: tel2 = l2.get_telemetry()
                except Exception: pass

        if tel1 is not None and tel2 is not None and "X" in tel1.columns and "Distance" in tel1.columns:
            num_minisectors = 25
            total_dist = max(tel1['Distance'].max(), tel2['Distance'].max())
            sector_length = total_dist / num_minisectors

            tel1['Minisector'] = (tel1['Distance'] // sector_length).astype(int)
            tel2['Minisector'] = (tel2['Distance'] // sector_length).astype(int)

            d1_avg = tel1.groupby('Minisector')['Speed'].mean()
            d2_avg = tel2.groupby('Minisector')['Speed'].mean()

            faster_driver = {}
            for sector in range(num_minisectors):
                s1 = d1_avg.get(sector, 0)
                s2 = d2_avg.get(sector, 0)
                faster_driver[sector] = dom_d1 if s1 >= s2 else dom_d2

            tel1['FasterDriver'] = tel1['Minisector'].map(faster_driver)

            palette = {dom_d1: "#FF1801", dom_d2: "#00D2BE"}
            fig_dom = px.scatter(
                tel1,
                x="X",
                y="Y",
                color="FasterDriver",
                color_discrete_map=palette,
                title=f"Minisector Track Dominance: {dom_d1} vs {dom_d2}"
            )
            fig_dom.update_yaxes(scaleanchor="x", scaleratio=1)
            fig_dom.update_traces(marker=dict(size=4))
            fig_dom.update_layout(
                template="plotly_dark",
                xaxis=dict(showgrid=False, zeroline=False, visible=False),
                yaxis=dict(showgrid=False, zeroline=False, visible=False),
                height=700
            )
            st.plotly_chart(fig_dom, use_container_width=True)
        else:
            st.warning("Insufficient telemetry or GPS data for one or both drivers.")


# 2. STINTS & TIRES
with tab_stints:
    st.subheader("Driver Stint Strategies & Tire Wear")

    selected_stint_drivers = st.multiselect(
        "Select Drivers to Display",
        options=drivers,
        default=drivers
    )

    if not selected_stint_drivers:
        st.info("Please select at least one driver to display stints.")
    else:
        stint_cols = ["Driver", "Stint", "Compound", "LapNumber", "TyreLife"]
        available_stint_cols = [c for c in stint_cols if c in session.laps.columns]
        stints_raw = session.laps[available_stint_cols].dropna(subset=["Stint", "Compound"]).copy()
        stints_raw = stints_raw[stints_raw["Driver"].isin(selected_stint_drivers)]

        agg_dict = {"StintLength": ("LapNumber", "count")}
        if "TyreLife" in stints_raw.columns:
            agg_dict["StartTyreLife"] = ("TyreLife", "min")
            agg_dict["EndTyreLife"] = ("TyreLife", "max")

        stints = stints_raw.groupby(["Driver", "Stint", "Compound"]).agg(**agg_dict).reset_index()

        if "StartTyreLife" in stints.columns:
            stints["StartTyreLife"] = stints["StartTyreLife"].round().astype("Int64")
            stints["EndTyreLife"] = stints["EndTyreLife"].round().astype("Int64")

        compound_colors = {
            "SOFT": "#FF3333",
            "MEDIUM": "#FFF200",
            "HARD": "#FFFFFF",
            "INTERMEDIATE": "#39B54A",
            "WET": "#00AEEF"
        }

        calculated_height = max(350, len(selected_stint_drivers) * 35)

        fig_stint = px.bar(
            stints,
            y="Driver",
            x="StintLength",
            color="Compound",
            orientation="h",
            title="Laps Driven Per Stint (Hover to view Tire Age & Wear)",
            color_discrete_map=compound_colors,
            height=calculated_height
        )

        fig_stint.update_yaxes(type='category', dtick=1, categoryorder='total ascending')
        fig_stint.update_layout(template="plotly_dark", barmode="stack")

        st.plotly_chart(fig_stint, use_container_width=True)

        with st.expander("📋 View Detailed Stint & Tire Wear Data Table", expanded=False):
            st.dataframe(stints, use_container_width=True, hide_index=True)


# 3. CAR TELEMETRY
with tab_telemetry:
    st.subheader("Car Telemetry Analysis")

    session_fastest = session.laps.pick_fastest()
    if session_fastest is not None:
        fl_driver = session_fastest['Driver']
        fl_time_str = str(session_fastest['LapTime']).split()[-1][:12]
        fl_lap_num = int(session_fastest['LapNumber'])
        fl_compound = session_fastest.get('Compound', 'N/A')

        fl_col1, fl_col2, fl_col3 = st.columns([1.5, 1, 1])
        fl_col1.metric("🟣 Overall Fastest Lap", f"{fl_driver} ({fl_time_str})")
        fl_col2.metric("Lap Number", f"Lap {fl_lap_num}")
        fl_col3.metric("Tire Compound", fl_compound)
        st.divider()
    else:
        fl_driver = drivers[0]

    col_mode, col_d1, col_d2 = st.columns([1, 1, 1])
    with col_mode:
        lap_mode = st.radio("Lap Selection Mode", ["Fastest Lap", "Select by Lap Number"])

    with col_d1:
        default_idx = drivers.index(fl_driver) if fl_driver in drivers else 0
        driver_1 = st.selectbox("Primary Driver", drivers, index=default_idx)

    with col_d2:
        compare_mode = st.checkbox("Compare with another driver", value=False)
        driver_2 = st.selectbox("Comparison Driver", drivers, index=1 if len(drivers) > 1 else 0, disabled=not compare_mode)

    selected_lap_num = None
    if lap_mode == "Select by Lap Number":
        max_laps = int(session.laps['LapNumber'].max())
        selected_lap_num = st.slider("Select Lap Number", min_value=1, max_value=max_laps, value=1)

    def get_telemetry_for_driver(driver_code):
        if hasattr(session, "get_driver_telemetry") and session.get_driver_telemetry(driver_code) is not None:
            laps = session.laps.pick_driver(driver_code)
            lap = laps.pick_fastest() if not laps.empty else None
            return lap, session.get_driver_telemetry(driver_code)

        laps = session.laps.pick_driver(driver_code)
        if laps.empty: return None, None
        
        if lap_mode == "Fastest Lap":
            lap = laps.pick_fastest()
        else:
            match = laps[laps['LapNumber'] == selected_lap_num]
            if match.empty: return None, None
            lap = match.iloc[0]

        try:
            return lap, lap.get_telemetry()
        except Exception:
            return lap, None

    lap_1, tel_1 = get_telemetry_for_driver(driver_1)

    if tel_1 is not None and not tel_1.empty:
        lap_desc_1 = f"Lap {lap_1['LapNumber']} ({lap_1['LapTime']})" if lap_1 is not None and pd.notna(lap_1['LapTime']) else "Fastest Lap"

        if compare_mode:
            lap_2, tel_2 = get_telemetry_for_driver(driver_2)
            
            if tel_2 is not None and not tel_2.empty:
                lap_desc_2 = f"Lap {lap_2['LapNumber']} ({lap_2['LapTime']})" if lap_2 is not None and pd.notna(lap_2['LapTime']) else "Fastest Lap"

                fig_tel = sp.make_subplots(
                    rows=5, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    row_heights=[0.35, 0.18, 0.12, 0.15, 0.20],
                    subplot_titles=(
                        f"Speed Comparison: {driver_1} vs {driver_2}",
                        "Throttle (%)",
                        "Brake Application",
                        "Gear Selection",
                        f"Time Delta (Negative = {driver_1} ahead | Positive = {driver_2} ahead)"
                    )
                )

                c1, c2 = "#FF1801", "#00D2BE"

                fig_tel.add_trace(go.Scatter(x=tel_1["Distance"], y=tel_1["Speed"], name=f"{driver_1} - {lap_desc_1}", line=dict(color=c1, width=2)), row=1, col=1)
                fig_tel.add_trace(go.Scatter(x=tel_2["Distance"], y=tel_2["Speed"], name=f"{driver_2} - {lap_desc_2}", line=dict(color=c2, width=2)), row=1, col=1)

                fig_tel.add_trace(go.Scatter(x=tel_1["Distance"], y=tel_1["Throttle"], name=f"{driver_1} Throttle", line=dict(color=c1, width=1.5), showlegend=False), row=2, col=1)
                fig_tel.add_trace(go.Scatter(x=tel_2["Distance"], y=tel_2["Throttle"], name=f"{driver_2} Throttle", line=dict(color=c2, width=1.5), showlegend=False), row=2, col=1)

                b1 = tel_1["Brake"].astype(int) if tel_1["Brake"].dtype == bool else tel_1["Brake"]
                b2 = tel_2["Brake"].astype(int) if tel_2["Brake"].dtype == bool else tel_2["Brake"]

                fig_tel.add_trace(go.Scatter(x=tel_1["Distance"], y=b1, name=f"{driver_1} Brake", line=dict(color=c1, width=1.5), fill="tozeroy", showlegend=False), row=3, col=1)
                fig_tel.add_trace(go.Scatter(x=tel_2["Distance"], y=b2, name=f"{driver_2} Brake", line=dict(color=c2, width=1.5), fill="tozeroy", showlegend=False), row=3, col=1)

                fig_tel.add_trace(go.Scatter(x=tel_1["Distance"], y=tel_1["nGear"], name=f"{driver_1} Gear", line=dict(color=c1, width=1.5, shape='hv'), showlegend=False), row=4, col=1)
                fig_tel.add_trace(go.Scatter(x=tel_2["Distance"], y=tel_2["nGear"], name=f"{driver_2} Gear", line=dict(color=c2, width=1.5, shape='hv'), showlegend=False), row=4, col=1)

                try:
                    delta_time, ref_tel, _ = utils.delta_time(lap_1, lap_2)
                    delta_dist = ref_tel["Distance"]
                except Exception:
                    t1_sec = pd.to_timedelta(tel_1['Time']).dt.total_seconds().values
                    t2_sec = pd.to_timedelta(tel_2['Time']).dt.total_seconds().values
                    common_dist = tel_1['Distance'].values
                    t2_resampled = np.interp(common_dist, tel_2['Distance'].values, t2_sec)
                    delta_time = t1_sec - t2_resampled
                    delta_dist = common_dist

                fig_tel.add_trace(go.Scatter(x=delta_dist, y=delta_time, name="Delta-T", line=dict(color="#FFFFFF", width=1.5), fill="tozeroy", showlegend=False), row=5, col=1)

                fig_tel.update_yaxes(title_text="Speed (km/h)", row=1, col=1)
                fig_tel.update_yaxes(title_text="Throttle (%)", range=[-5, 105], row=2, col=1)
                fig_tel.update_yaxes(title_text="Brake", range=[-0.1, 1.1] if b1.max() <= 1 else [-5, 105], row=3, col=1)
                fig_tel.update_yaxes(title_text="Gear", dtick=1, range=[0.5, 8.5], row=4, col=1)
                fig_tel.update_yaxes(title_text="Delta (s)", row=5, col=1)
                fig_tel.update_xaxes(title_text="Track Distance (m)", row=5, col=1)

                fig_tel.update_layout(template="plotly_dark", height=1000, hovermode="x unified")
                st.plotly_chart(fig_tel, use_container_width=True)
            else:
                st.warning(f"No telemetry data found for {driver_2}.")
        else:
            fig_single = sp.make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.04,
                row_heights=[0.45, 0.20, 0.15, 0.20],
                subplot_titles=(
                    f"Speed Trace: {driver_1} ({lap_desc_1})",
                    "Throttle (%)",
                    "Brake Application",
                    "Gear Selection"
                )
            )

            fig_single.add_trace(go.Scatter(x=tel_1["Distance"], y=tel_1["Speed"], name="Speed", line=dict(color="#FF1801", width=2)), row=1, col=1)
            fig_single.add_trace(go.Scatter(x=tel_1["Distance"], y=tel_1["Throttle"], name="Throttle", line=dict(color="#00D2BE", width=1.5)), row=2, col=1)
            b1 = tel_1["Brake"].astype(int) if tel_1["Brake"].dtype == bool else tel_1["Brake"]
            fig_single.add_trace(go.Scatter(x=tel_1["Distance"], y=b1, name="Brake", line=dict(color="#FF3333", width=1.5), fill="tozeroy"), row=3, col=1)
            fig_single.add_trace(go.Scatter(x=tel_1["Distance"], y=tel_1["nGear"], name="Gear", line=dict(color="#FFF200", width=1.5, shape='hv')), row=4, col=1)

            fig_single.update_yaxes(title_text="km/h", row=1, col=1)
            fig_single.update_yaxes(title_text="Throttle %", range=[-5, 105], row=2, col=1)
            fig_single.update_yaxes(title_text="Brake", range=[-0.1, 1.1] if b1.max() <= 1 else [-5, 105], row=3, col=1)
            fig_single.update_yaxes(title_text="Gear", dtick=1, range=[0.5, 8.5], row=4, col=1)
            fig_single.update_xaxes(title_text="Track Distance (m)", row=4, col=1)

            fig_single.update_layout(template="plotly_dark", height=850, hovermode="x unified", showlegend=False)
            st.plotly_chart(fig_single, use_container_width=True)
    else:
        st.warning(f"No telemetry data found for {driver_1}.")


# 4. RACE PROGRESSION
with tab_positions:
    st.subheader("Lap-by-Lap Position Changes")
    pos_data = session.laps[['LapNumber', 'Driver', 'Position']].dropna().copy()
    
    selected_drivers = st.multiselect("Highlight Drivers", drivers, default=drivers[:8] if len(drivers) >= 8 else drivers)
    filtered_pos = pos_data[pos_data['Driver'].isin(selected_drivers)]
    
    fig_bump = px.line(filtered_pos, x="LapNumber", y="Position", color="Driver", markers=True, title=f"Position Changes: {grand_prix} ({year})")
    fig_bump.update_yaxes(autorange="reversed", dtick=1)
    fig_bump.update_xaxes(dtick=5)
    fig_bump.update_layout(template="plotly_dark", height=650)
    st.plotly_chart(fig_bump, use_container_width=True)


# 5. TIRE DEGRADATION ML MODEL
with tab_deg:
    st.subheader("Machine Learning: Tire Degradation & Stint Pace Predictor")
    st.caption("Fits an Ordinary Least Squares (OLS) Regression model on representative laps to isolate wear rate and predict stint cliff.")

    col_ml_d, col_ml_stint = st.columns(2)
    with col_ml_d:
        deg_driver = st.selectbox("Select Driver to Model", drivers, index=0)

    driver_all_laps = session.laps.pick_driver(deg_driver)
    
    if not driver_all_laps.empty:
        available_stints = sorted(driver_all_laps['Stint'].dropna().unique())
        with col_ml_stint:
            chosen_stint = st.selectbox("Select Stint Number", available_stints, index=0)

        stint_laps = driver_all_laps[driver_all_laps['Stint'] == chosen_stint].copy()
        try:
            stint_laps = stint_laps.pick_quicklaps().dropna(subset=['TyreLife', 'LapTime'])
        except Exception:
            stint_laps = stint_laps.dropna(subset=['TyreLife', 'LapTime'])

        if len(stint_laps) >= 4:
            compound_used = stint_laps['Compound'].iloc[0]
            stint_laps['LapTimeSeconds'] = stint_laps['LapTime'].dt.total_seconds()

            X = stint_laps[['TyreLife']].values
            y = stint_laps['LapTimeSeconds'].values

            model = LinearRegression()
            model.fit(X, y)

            deg_per_lap = model.coef_[0]
            base_pace = model.intercept_
            predictions = model.predict(X)
            r2 = r2_score(y, predictions)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Tire Compound", compound_used)
            m2.metric("Base Pace (Fresh Tire)", f"{base_pace:.3f} s")
            m3.metric("Degradation Rate", f"+{deg_per_lap:.3f} s/lap" if deg_per_lap > 0 else f"{deg_per_lap:.3f} s/lap")
            m4.metric("Model Fit ($R^2$)", f"{r2:.2f}")

            fig_ml = go.Figure()
            fig_ml.add_trace(go.Scatter(
                x=stint_laps['TyreLife'], y=stint_laps['LapTimeSeconds'],
                mode='markers', name='Actual Lap Time',
                marker=dict(size=9, color="#00D2BE", line=dict(width=1, color="white"))
            ))

            line_x = np.linspace(float(X.min()), float(X.max()), 50).reshape(-1, 1)
            line_y = model.predict(line_x)

            fig_ml.add_trace(go.Scatter(
                x=line_x.flatten(), y=line_y,
                mode='lines', name=f'Fit: +{deg_per_lap:.3f}s / lap',
                line=dict(color="#FF1801", width=2.5, dash='dash')
            ))

            fig_ml.update_layout(
                title=f"{deg_driver} Stint {chosen_stint} ({compound_used} Tires) Degradation Profile",
                xaxis_title="Tire Life (Laps)",
                yaxis_title="Lap Time (Seconds)",
                template="plotly_dark",
                height=520
            )
            st.plotly_chart(fig_ml, use_container_width=True)

            st.markdown("#### Stint Extrapolation Simulator")
            current_max_life = int(X.max())
            sim_laps = st.slider(
                "Simulate pace if stint extended to tire age:", 
                min_value=current_max_life + 1, 
                max_value=current_max_life + 20, 
                value=current_max_life + 5
            )
            
            projected_lap_time = model.predict([[sim_laps]])[0]
            delta_from_base = projected_lap_time - base_pace
            
            st.info(
                f"💡 At **Lap {sim_laps}** on this set, projected pace is **{projected_lap_time:.3f} s** "
                f"(a drop-off of **{delta_from_base:.2f} s** compared to fresh rubber)."
            )
        else:
            st.warning("Not enough clean laps in this stint to train the regression model (minimum 4 laps required).")
    else:
        st.warning(f"No laps found for {deg_driver}.")


# 6. CHAMPIONSHIP STANDINGS
with tab_standings:
    st.subheader(f"Championship Standings after {grand_prix} ({year})")
    round_num = session.event.get('RoundNumber', 1) if hasattr(session, 'event') else 1
    
    df_wdc, df_wcc = get_standings_after_race(year, round_num)
    
    # Offline Fallback: compute directly from session.results if API call misses
    if df_wdc is None or df_wcc is None or df_wdc.empty:
        if hasattr(session, "results") and not session.results.empty:
            res = session.results.copy()
            # Standard F1 race points scale
            pts_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
            
            if 'Points' not in res.columns or res['Points'].sum() == 0:
                res['Points'] = res['Position'].map(pts_map).fillna(0)
            
            # Construct WDC
            wdc_list = []
            for _, r in res.iterrows():
                wdc_list.append({
                    "Pos": int(r['Position']) if pd.notna(r['Position']) else 99,
                    "Driver": r.get('FullName', r.get('Abbreviation', 'Unknown')),
                    "Team": r.get('TeamName', 'N/A'),
                    "Points": float(r.get('Points', 0)),
                    "Wins": 1 if r.get('Position') == 1 else 0
                })
            df_wdc = pd.DataFrame(wdc_list).sort_values(by="Pos", ascending=True)

            # Construct WCC
            wcc_df = res.groupby('TeamName')['Points'].sum().reset_index()
            wcc_df.columns = ['Team', 'Points']
            wcc_df['Wins'] = res[res['Position'] == 1]['TeamName'].value_counts().reindex(wcc_df['Team'], fill_value=0).values
            wcc_df = wcc_df.sort_values(by="Points", ascending=False).reset_index(drop=True)
            wcc_df['Pos'] = wcc_df.index + 1
            df_wcc = wcc_df[['Pos', 'Team', 'Points', 'Wins']]

    if df_wdc is not None and df_wcc is not None and not df_wdc.empty:
        col_wdc, col_wcc = st.columns(2)
        with col_wdc:
            st.markdown("#### 🏎️ Drivers' Championship (WDC)")
            fig_wdc = px.bar(
                df_wdc.sort_values(by="Points", ascending=True),
                x="Points", y="Driver", orientation="h", text="Points",
                title="WDC Points", height=650, color="Points", color_continuous_scale="Purples"
            )
            fig_wdc.update_layout(template="plotly_dark", showlegend=False)
            st.plotly_chart(fig_wdc, use_container_width=True)
            with st.expander("View Full WDC Table"):
                st.dataframe(df_wdc[["Pos", "Driver", "Team", "Points", "Wins"]], hide_index=True, use_container_width=True)

        with col_wcc:
            st.markdown("#### 🛠️ Constructors' Championship (WCC)")
            fig_wcc = px.bar(
                df_wcc.sort_values(by="Points", ascending=True),
                x="Points", y="Team", orientation="h", text="Points",
                title="WCC Points", height=500, color="Points", color_continuous_scale="Reds"
            )
            fig_wcc.update_layout(template="plotly_dark", showlegend=False)
            st.plotly_chart(fig_wcc, use_container_width=True)
            with st.expander("View Full WCC Table"):
                st.dataframe(df_wcc[["Pos", "Team", "Points", "Wins"]], hide_index=True, use_container_width=True)
    else:
        st.info("Championship standings are unavailable for this round.")