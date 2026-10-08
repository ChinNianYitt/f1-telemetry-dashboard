import os
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
import tempfile


# --- FASTF1 CACHE SETUP ---
# Uses the system temp directory so 0 MB of permanent space is taken on your laptop
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
        # Query Jolpica / Ergast mirror for driver standings after this round
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
    schedule = fastf1.get_event_schedule(selected_year)
    # Filter out testing sessions to keep official rounds only
    gp_events = schedule[schedule['RoundNumber'] > 0].copy()
    
    event_names = gp_events['EventName'].tolist()
    
    # Compare event date/session date with current UTC timestamp
    now = pd.Timestamp.now(tz='UTC')
    
    # Convert schedule dates to UTC-aware timestamps
    event_dates = pd.to_datetime(gp_events['EventDate'], utc=True)
    past_events = gp_events[event_dates <= now]

    if not past_events.empty:
        # The last event in past_events is the most recently completed race
        latest_past_event_name = past_events.iloc[-1]['EventName']
        default_index = event_names.index(latest_past_event_name)
    else:
        # If no events have happened yet in this season, fallback to Round 1
        default_index = 0

    return event_names, default_index

# Fetch all events for the selected year
# Fetch events and determine the most recent race index
available_gps, default_gp_idx = get_season_calendar_and_default(year)

grand_prix = st.sidebar.selectbox(
    "Grand Prix", 
    available_gps, 
    index=default_gp_idx
)

# Grand Prix selector populates automatically based on the year chosen
session_type = st.sidebar.selectbox("Session", ["R", "Q"], format_func=lambda x: "Race" if x == "R" else "Qualifying")

@st.cache_data
def load_session(year, gp, session_code):
    try:
        session = fastf1.get_session(year, gp, session_code)
        session.load(telemetry=True, laps=True, weather=False)
        return session, None
    except Exception as e:
        return None, str(e)

with st.spinner("Fetching F1 session data..."):
    session, load_error = load_session(year, grand_prix, session_type)

# Safe check without triggering DataNotLoadedError
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
    st.stop()  # Cleanly halts execution without traceback

drivers = sorted(session.laps['Driver'].dropna().unique().tolist())
selected_driver = st.sidebar.selectbox("Select Primary Driver", drivers, index=0)



# --- TAB LAYOUT ---
tab_track, tab_stints, tab_telemetry, tab_positions, tab_deg, tab_standings, tab_teammates = st.tabs(["🗺️ Track Map", "🛞 Stints & Tires", "📈 Speed Telemetry", "🏁 Race Progression", "🤖 Tire Degradation (ML)", "🏆 Standings", "👥 Teammate Battles"])



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
            tel = fastest_lap.get_telemetry()

            # Render coordinates colored by Speed
            fig_heatmap = px.scatter(
                tel,
                x="X",
                y="Y",
                color="Speed",
                color_continuous_scale="Turbo",
                title=f"Speed Heatmap: {map_driver} ({fastest_lap['LapTime']})",
                labels={"Speed": "km/h"}
            )
            # Maintain 1:1 track geometry so turns aren't distorted
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
            st.warning(f"No lap data available for {map_driver}.")

    else:
        # HEAD-TO-HEAD TRACK DOMINANCE
        col_m1, col_m2 = st.columns(2)
        dom_d1 = col_m1.selectbox("Driver 1 (Dominance)", drivers, index=0)
        dom_d2 = col_m2.selectbox("Driver 2 (Dominance)", drivers, index=1 if len(drivers) > 1 else 0)

        d1_laps = session.laps.pick_driver(dom_d1)
        d2_laps = session.laps.pick_driver(dom_d2)

        if not d1_laps.empty and not d2_laps.empty:
            tel1 = d1_laps.pick_fastest().get_telemetry()
            tel2 = d2_laps.pick_fastest().get_telemetry()

            # Merge and resample by distance slices (every 25 minisectors)
            num_minisectors = 25
            total_dist = max(tel1['Distance'].max(), tel2['Distance'].max())
            sector_length = total_dist / num_minisectors

            # Assign minisector numbers based on distance
            tel1['Minisector'] = (tel1['Distance'] // sector_length).astype(int)
            tel2['Minisector'] = (tel2['Distance'] // sector_length).astype(int)

            # Calculate average speed per minisector
            d1_avg = tel1.groupby('Minisector')['Speed'].mean()
            d2_avg = tel2.groupby('Minisector')['Speed'].mean()

            # Determine who was faster in each minisector
            faster_driver = {}
            for sector in range(num_minisectors):
                s1 = d1_avg.get(sector, 0)
                s2 = d2_avg.get(sector, 0)
                faster_driver[sector] = dom_d1 if s1 >= s2 else dom_d2

            tel1['FasterDriver'] = tel1['Minisector'].map(faster_driver)

            # Plot coordinates colored by the dominant driver
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
            st.warning("Insufficient lap data for one or both selected drivers.")



# 2. STINTS & TIRES (With Driver Filter & Tire Wear)
with tab_stints:
    st.subheader("Driver Stint Strategies & Tire Wear")

    # Driver selector (defaults to all drivers)
    selected_stint_drivers = st.multiselect(
        "Select Drivers to Display",
        options=drivers,
        default=drivers
    )

    # If user deselects everything, give a hint
    if not selected_stint_drivers:
        st.info("Please select at least one driver to display stints.")
    else:
        # Extract stint data including TyreLife
        stint_cols = ["Driver", "Stint", "Compound", "LapNumber", "TyreLife"]
        stints_raw = session.laps[stint_cols].dropna(subset=["Stint", "Compound"]).copy()

        # Filter for the chosen drivers
        stints_raw = stints_raw[stints_raw["Driver"].isin(selected_stint_drivers)]

        # Aggregate stint metrics: Stint length, starting tyre life, and ending tyre life
        stints = stints_raw.groupby(["Driver", "Stint", "Compound"]).agg(
            StintLength=("LapNumber", "count"),
            StartTyreLife=("TyreLife", "min"),
            EndTyreLife=("TyreLife", "max")
        ).reset_index()

        # Safely convert to nullable integer
        stints["StartTyreLife"] = stints["StartTyreLife"].round().astype("Int64")
        stints["EndTyreLife"] = stints["EndTyreLife"].round().astype("Int64")

        compound_colors = {
            "SOFT": "#FF3333",
            "MEDIUM": "#FFF200",
            "HARD": "#FFFFFF",
            "INTERMEDIATE": "#39B54A",
            "WET": "#00AEEF"
        }

        # Dynamically scale chart height according to how many drivers are chosen
        calculated_height = max(350, len(selected_stint_drivers) * 35)

        fig_stint = px.bar(
            stints,
            y="Driver",
            x="StintLength",
            color="Compound",
            orientation="h",
            title="Laps Driven Per Stint (Hover to view Tire Age & Wear)",
            color_discrete_map=compound_colors,
            height=calculated_height,
            hover_data={
                "Driver": True,
                "Stint": True,
                "Compound": True,
                "StintLength": True,
                "StartTyreLife": True,
                "EndTyreLife": True
            }
        )

        fig_stint.update_yaxes(type='category', dtick=1, categoryorder='total ascending')
        fig_stint.update_layout(template="plotly_dark", barmode="stack")

        st.plotly_chart(fig_stint, use_container_width=True)

        # Detailed table view
        with st.expander("📋 View Detailed Stint & Tire Wear Data Table", expanded=False):
            st.dataframe(
                stints.rename(columns={
                    "StintLength": "Laps Run",
                    "StartTyreLife": "Starting Tire Age (Laps)",
                    "EndTyreLife": "Ending Tire Age (Laps)"
                }),
                use_container_width=True,
                hide_index=True
            )



# 3. CAR TELEMETRY (Single Driver or Comparison with Lap Selector & Delta Time)
with tab_telemetry:
    st.subheader("Car Telemetry Analysis")

    # --- FASTEST LAP OF THE ENTIRE SESSION ---
    session_fastest = session.laps.pick_fastest()
    fl_driver = session_fastest['Driver']
    fl_time_str = str(session_fastest['LapTime']).split()[-1][:12]  # Clean format: '01:21.046'
    fl_lap_num = int(session_fastest['LapNumber'])
    fl_compound = session_fastest['Compound']

    # Display clean banner metrics at the top
    fl_col1, fl_col2, fl_col3 = st.columns([1.5, 1, 1])
    fl_col1.metric("🟣 Overall Fastest Lap", f"{fl_driver} ({fl_time_str})")
    fl_col2.metric("Lap Number", f"Lap {fl_lap_num}")
    fl_col3.metric("Tire Compound", fl_compound)

    st.divider()

    # --- Mode & Driver Selectors ---
    col_mode, col_d1, col_d2 = st.columns([1, 1, 1])
    
    with col_mode:
        lap_mode = st.radio(
            "Lap Selection Mode",
            ["Fastest Lap", "Select by Lap Number"],
            horizontal=False
        )

    with col_d1:
        # Default driver index: picks the fastest lap driver automatically if in the list
        default_idx = drivers.index(fl_driver) if fl_driver in drivers else 0
        driver_1 = st.selectbox("Primary Driver", drivers, index=default_idx)

    with col_d2:
        compare_mode = st.checkbox("Compare with another driver", value=False)
        driver_2 = st.selectbox(
            "Comparison Driver", 
            drivers, 
            index=1 if len(drivers) > 1 else 0, 
            disabled=not compare_mode
        )

    # Lap number slider if manual mode is chosen
    selected_lap_num = None
    if lap_mode == "Select by Lap Number":
        max_laps = int(session.laps['LapNumber'].max())
        selected_lap_num = st.slider("Select Lap Number", min_value=1, max_value=max_laps, value=1)

    # Helper function to get telemetry based on mode
    def get_driver_telemetry(driver_code):
        laps = session.laps.pick_driver(driver_code)
        if laps.empty:
            return None, None
        
        if lap_mode == "Fastest Lap":
            lap = laps.pick_fastest()
        else:
            match = laps[laps['LapNumber'] == selected_lap_num]
            if match.empty:
                return None, None
            lap = match.iloc[0]

        # Ensure telemetry exists for this specific lap
        try:
            telemetry = lap.get_telemetry()
            return lap, telemetry
        except Exception:
            return lap, None

    lap_1, tel_1 = get_driver_telemetry(driver_1)

    if tel_1 is not None:
        lap_desc_1 = f"Lap {lap_1['LapNumber']} ({lap_1['LapTime']})" if pd.notna(lap_1['LapTime']) else f"Lap {lap_1['LapNumber']}"

        if compare_mode:
            lap_2, tel_2 = get_driver_telemetry(driver_2)
            
            if tel_2 is not None:
                lap_desc_2 = f"Lap {lap_2['LapNumber']} ({lap_2['LapTime']})" if pd.notna(lap_2['LapTime']) else f"Lap {lap_2['LapNumber']}"

                # 5-Row Subplot: Speed, Throttle, Brake, Gear, Time Delta
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

                c1 = "#FF1801"  # Driver 1 (Red)
                c2 = "#00D2BE"  # Driver 2 (Teal)

                # --- 1. Speed Traces ---
                fig_tel.add_trace(go.Scatter(
                    x=tel_1["Distance"], y=tel_1["Speed"],
                    name=f"{driver_1} - {lap_desc_1}",
                    line=dict(color=c1, width=2)
                ), row=1, col=1)

                fig_tel.add_trace(go.Scatter(
                    x=tel_2["Distance"], y=tel_2["Speed"],
                    name=f"{driver_2} - {lap_desc_2}",
                    line=dict(color=c2, width=2)
                ), row=1, col=1)

                # --- 2. Throttle Traces ---
                fig_tel.add_trace(go.Scatter(
                    x=tel_1["Distance"], y=tel_1["Throttle"],
                    name=f"{driver_1} Throttle",
                    line=dict(color=c1, width=1.5),
                    showlegend=False
                ), row=2, col=1)

                fig_tel.add_trace(go.Scatter(
                    x=tel_2["Distance"], y=tel_2["Throttle"],
                    name=f"{driver_2} Throttle",
                    line=dict(color=c2, width=1.5),
                    showlegend=False
                ), row=2, col=1)

                # --- 3. Brake Traces ---
                b1 = tel_1["Brake"].astype(int) if tel_1["Brake"].dtype == bool else tel_1["Brake"]
                b2 = tel_2["Brake"].astype(int) if tel_2["Brake"].dtype == bool else tel_2["Brake"]

                fig_tel.add_trace(go.Scatter(
                    x=tel_1["Distance"], y=b1,
                    name=f"{driver_1} Brake",
                    line=dict(color=c1, width=1.5),
                    fill="tozeroy",
                    showlegend=False
                ), row=3, col=1)

                fig_tel.add_trace(go.Scatter(
                    x=tel_2["Distance"], y=b2,
                    name=f"{driver_2} Brake",
                    line=dict(color=c2, width=1.5),
                    fill="tozeroy",
                    showlegend=False
                ), row=3, col=1)

                # --- 4. Gear Traces ---
                fig_tel.add_trace(go.Scatter(
                    x=tel_1["Distance"], y=tel_1["nGear"],
                    name=f"{driver_1} Gear",
                    line=dict(color=c1, width=1.5, shape='hv'),
                    showlegend=False
                ), row=4, col=1)

                fig_tel.add_trace(go.Scatter(
                    x=tel_2["Distance"], y=tel_2["nGear"],
                    name=f"{driver_2} Gear",
                    line=dict(color=c2, width=1.5, shape='hv'),
                    showlegend=False
                ), row=4, col=1)

                # --- 5. Time Delta Calculation ---
                try:
                    from fastf1 import utils
                    delta_time, ref_tel, _ = utils.delta_time(lap_1, lap_2)
                    delta_dist = ref_tel["Distance"]
                except Exception:
                    t1_interp = pd.Series(tel_1['Time'].dt.total_seconds().values, index=tel_1['Distance'])
                    t2_interp = pd.Series(tel_2['Time'].dt.total_seconds().values, index=tel_2['Distance'])
                    common_dist = tel_1['Distance']
                    t2_resampled = np.interp(common_dist, tel_2['Distance'], t2_interp)
                    delta_time = t1_interp.values - t2_resampled
                    delta_dist = common_dist

                fig_tel.add_trace(go.Scatter(
                    x=delta_dist, y=delta_time,
                    name="Delta-T",
                    line=dict(color="#FFFFFF", width=1.5),
                    fill="tozeroy",
                    showlegend=False
                ), row=5, col=1)

                # Axis & Layout Formatting
                fig_tel.update_yaxes(title_text="Speed (km/h)", row=1, col=1)
                fig_tel.update_yaxes(title_text="Throttle (%)", range=[-5, 105], row=2, col=1)
                fig_tel.update_yaxes(title_text="Brake", range=[-0.1, 1.1] if b1.max() <= 1 else [-5, 105], row=3, col=1)
                fig_tel.update_yaxes(title_text="Gear", dtick=1, range=[0.5, 8.5], row=4, col=1)
                fig_tel.update_yaxes(title_text="Delta (s)", row=5, col=1)
                fig_tel.update_xaxes(title_text="Track Distance (m)", row=5, col=1)

                fig_tel.update_layout(
                    template="plotly_dark",
                    height=1000,
                    hovermode="x unified"
                )

                st.plotly_chart(fig_tel, use_container_width=True)
            else:
                st.warning(f"No telemetry data found for {driver_2} on the selected lap.")
        else:
            # Single Driver Full Telemetry Stack (Speed, Throttle, Brake, Gear)
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

            # Speed
            fig_single.add_trace(go.Scatter(
                x=tel_1["Distance"], y=tel_1["Speed"],
                name="Speed", line=dict(color="#FF1801", width=2)
            ), row=1, col=1)

            # Throttle
            fig_single.add_trace(go.Scatter(
                x=tel_1["Distance"], y=tel_1["Throttle"],
                name="Throttle", line=dict(color="#00D2BE", width=1.5)
            ), row=2, col=1)

            # Brake
            b1 = tel_1["Brake"].astype(int) if tel_1["Brake"].dtype == bool else tel_1["Brake"]
            fig_single.add_trace(go.Scatter(
                x=tel_1["Distance"], y=b1,
                name="Brake", line=dict(color="#FF3333", width=1.5), fill="tozeroy"
            ), row=3, col=1)

            # Gear
            fig_single.add_trace(go.Scatter(
                x=tel_1["Distance"], y=tel_1["nGear"],
                name="Gear", line=dict(color="#FFF200", width=1.5, shape='hv')
            ), row=4, col=1)

            fig_single.update_yaxes(title_text="km/h", row=1, col=1)
            fig_single.update_yaxes(title_text="Throttle %", range=[-5, 105], row=2, col=1)
            fig_single.update_yaxes(title_text="Brake", range=[-0.1, 1.1] if b1.max() <= 1 else [-5, 105], row=3, col=1)
            fig_single.update_yaxes(title_text="Gear", dtick=1, range=[0.5, 8.5], row=4, col=1)
            fig_single.update_xaxes(title_text="Track Distance (m)", row=4, col=1)

            fig_single.update_layout(
                template="plotly_dark",
                height=850,
                hovermode="x unified",
                showlegend=False
            )

            st.plotly_chart(fig_single, use_container_width=True)
    else:
        st.warning(f"No telemetry data found for {driver_1} on the selected lap.")




# 4. RACE PROGRESSION (BUMP CHART)
with tab_positions:
    st.subheader("Lap-by-Lap Position Changes")
    
    # Filter laps with valid positions and lap numbers
    pos_data = session.laps[['LapNumber', 'Driver', 'Position']].dropna().copy()
    
    # Multi-select so the user can filter to a subset or keep all drivers
    selected_drivers = st.multiselect(
        "Highlight Drivers", 
        drivers, 
        default=drivers[:8]  # Defaults to top 8 drivers to keep it readable
    )
    
    filtered_pos = pos_data[pos_data['Driver'].isin(selected_drivers)]
    
    fig_bump = px.line(
        filtered_pos,
        x="LapNumber",
        y="Position",
        color="Driver",
        markers=True,
        title=f"Position Changes: {grand_prix} ({year})"
    )
    
    # Invert y-axis so P1 is at the top
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

    # Filter laps for chosen driver
    driver_all_laps = session.laps.pick_driver(deg_driver)
    
    if not driver_all_laps.empty:
        available_stints = sorted(driver_all_laps['Stint'].dropna().unique())
        
        with col_ml_stint:
            chosen_stint = st.selectbox("Select Stint Number", available_stints, index=0)

        # --- DATA SANITIZATION ---
        # 1. Filter for the selected stint
        stint_laps = driver_all_laps[driver_all_laps['Stint'] == chosen_stint].copy()
        
        # 2. Use pick_quicklaps() to eliminate pit in/out laps, safety car laps, and extreme traffic
        stint_laps = stint_laps.pick_quicklaps().dropna(subset=['TyreLife', 'LapTime'])

        if len(stint_laps) >= 4:
            compound_used = stint_laps['Compound'].iloc[0]
            stint_laps['LapTimeSeconds'] = stint_laps['LapTime'].dt.total_seconds()

            # --- MODEL TRAINING (SCIKIT-LEARN) ---
            # X: Feature matrix (Tyre Age in laps)
            # y: Target variable (Lap time in seconds)
            X = stint_laps[['TyreLife']].values
            y = stint_laps['LapTimeSeconds'].values

            model = LinearRegression()
            model.fit(X, y)

            deg_per_lap = model.coef_[0]          # Slope: Delta time per lap added
            base_pace = model.intercept_           # Intercept: Clean pace at Lap 0
            predictions = model.predict(X)
            r2 = r2_score(y, predictions)         # Goodness of fit

            # Metrics row
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Tire Compound", compound_used)
            m2.metric("Base Pace (Fresh Tire)", f"{base_pace:.3f} s")
            m3.metric("Degradation Rate", f"+{deg_per_lap:.3f} s/lap" if deg_per_lap > 0 else f"{deg_per_lap:.3f} s/lap")
            m4.metric("Model Fit ($R^2$)", f"{r2:.2f}")

            # --- VISUALIZATION ---
            fig_ml = go.Figure()

            # Actual lap times
            fig_ml.add_trace(go.Scatter(
                x=stint_laps['TyreLife'],
                y=stint_laps['LapTimeSeconds'],
                mode='markers',
                name='Actual Lap Time',
                marker=dict(size=9, color="#00D2BE", line=dict(width=1, color="white"))
            ))

            # Regression line
            line_x = np.linspace(float(X.min()), float(X.max()), 50).reshape(-1, 1)
            line_y = model.predict(line_x)

            fig_ml.add_trace(go.Scatter(
                x=line_x.flatten(),
                y=line_y,
                mode='lines',
                name=f'Fit: +{deg_per_lap:.3f}s / lap',
                line=dict(color="#FF1801", width=2.5, dash='dash')
            ))

            fig_ml.update_layout(
                title=f"{deg_driver} - Stint {chosen_stint} ({compound_used} Tires) Degradation Profile",
                xaxis_title="Tire Life (Laps)",
                yaxis_title="Lap Time (Seconds)",
                template="plotly_dark",
                height=520
            )

            st.plotly_chart(fig_ml, use_container_width=True)

            # --- STRATEGY SIMULATOR ---
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
            st.warning("Not enough clean laps in this stint to train the regression model (minimum 4 laps required after removing out-laps and safety car periods).")
    else:
        st.warning(f"No laps found for {deg_driver}.")


# 6. CHAMPIONSHIP STANDINGS TAB
with tab_standings:
    st.subheader(f"Championship Standings after {grand_prix} ({year})")
    
    # Retrieve the official round number from session event metadata
    round_num = session.event['RoundNumber']
    
    df_wdc, df_wcc = get_standings_after_race(year, round_num)
    
    if df_wdc is not None and df_wcc is not None:
        col_wdc, col_wcc = st.columns(2)
        
        with col_wdc:
            st.markdown("#### 🏎️ Drivers' Championship (WDC)")
            # Interactive points bar chart
            fig_wdc = px.bar(
                df_wdc.sort_values(by="Points", ascending=True),
                x="Points",
                y="Driver",
                orientation="h",
                text="Points",
                title="WDC Points",
                height=650,
                color="Points",
                color_continuous_scale="Purples"
            )
            fig_wdc.update_layout(template="plotly_dark", showlegend=False)
            st.plotly_chart(fig_wdc, use_container_width=True)
            
            with st.expander("View Full WDC Table"):
                st.dataframe(df_wdc[["Pos", "Driver", "Team", "Points", "Wins"]], hide_index=True, use_container_width=True)

        with col_wcc:
            st.markdown("#### 🛠️ Constructors' Championship (WCC)")
            # Interactive constructors bar chart
            fig_wcc = px.bar(
                df_wcc.sort_values(by="Points", ascending=True),
                x="Points",
                y="Team",
                orientation="h",
                text="Points",
                title="WCC Points",
                height=500,
                color="Points",
                color_continuous_scale="Reds"
            )
            fig_wcc.update_layout(template="plotly_dark", showlegend=False)
            st.plotly_chart(fig_wcc, use_container_width=True)
            
            with st.expander("View Full WCC Table"):
                st.dataframe(df_wcc[["Pos", "Team", "Points", "Wins"]], hide_index=True, use_container_width=True)
    else:
        st.info("Championship standings are unavailable for this round.")

# 7. TEAMMATE HEAD-TO-HEAD BREAKDOWN
with tab_teammates:
    st.subheader("Teammate Pace & Head-to-Head Benchmark")
    st.caption("Direct comparison between teammates running identical chassis, aero packages, and power units.")

    # 1. Map Teams to Drivers dynamically from session results
    results = session.results
    if not results.empty:
        # Group drivers by TeamName
        team_groups = {}
        for _, row in results.iterrows():
            team = row['TeamName']
            code = row['Abbreviation']
            if pd.notna(team) and pd.notna(code):
                team_groups.setdefault(team, []).append(code)

        # Filter for teams that have at least 2 drivers participating
        valid_teams = {k: v for k, v in team_groups.items() if len(v) >= 2}

        if valid_teams:
            selected_team = st.selectbox("Select Team Constructor", list(valid_teams.keys()), index=0)
            d1, d2 = valid_teams[selected_team][:2]

            # Fetch laps for each driver
            laps_d1 = session.laps.pick_driver(d1)
            laps_d2 = session.laps.pick_driver(d2)

            # Fastest single lap comparison
            fast_1 = laps_d1.pick_fastest()
            fast_2 = laps_d2.pick_fastest()

            # Clean racing laps (pick_quicklaps removes in/out/SC laps)
            quick_1 = laps_d1.pick_quicklaps().copy()
            quick_2 = laps_d2.pick_quicklaps().copy()

            # --- METRICS OVERVIEW ---
            col_m1, col_m2, col_m3 = st.columns(3)

            # 1. Single Lap Pace Gap
            if fast_1 is not None and fast_2 is not None:
                t1_sec = fast_1['LapTime'].total_seconds()
                t2_sec = fast_2['LapTime'].total_seconds()
                diff_single = t1_sec - t2_sec
                faster_driver = d1 if diff_single < 0 else d2
                margin = abs(diff_single)
                col_m1.metric("Single-Lap Edge", f"{faster_driver} (-{margin:.3f}s)")
            else:
                col_m1.metric("Single-Lap Edge", "N/A")

            # 2. Median True Race Pace Gap
            if not quick_1.empty and not quick_2.empty:
                med_1 = quick_1['LapTime'].dt.total_seconds().median()
                med_2 = quick_2['LapTime'].dt.total_seconds().median()
                diff_med = med_1 - med_2
                race_pace_leader = d1 if diff_med < 0 else d2
                race_margin = abs(diff_med)
                col_m2.metric("Median Race Pace", f"{race_pace_leader} (-{race_margin:.3f}s/lap)")
            else:
                col_m2.metric("Median Race Pace", "N/A")

            # 3. Position Finish / Status
            p1_finish = results.loc[results['Abbreviation'] == d1, 'Position'].values[0]
            p2_finish = results.loc[results['Abbreviation'] == d2, 'Position'].values[0]
            col_m3.metric("Final Finish (Classified)", f"{d1}: P{int(p1_finish) if pd.notna(p1_finish) else 'DNF'} | {d2}: P{int(p2_finish) if pd.notna(p2_finish) else 'DNF'}")

            st.divider()

            # --- SECTOR DELTA BREAKDOWN TABLE ---
            st.markdown("#### Sector Best Splits (Personal Best)")
            if fast_1 is not None and fast_2 is not None:
                # Calculate individual sector times
                s1_d1 = laps_d1['Sector1Time'].min().total_seconds() if pd.notna(laps_d1['Sector1Time'].min()) else None
                s1_d2 = laps_d2['Sector1Time'].min().total_seconds() if pd.notna(laps_d2['Sector1Time'].min()) else None

                s2_d1 = laps_d1['Sector2Time'].min().total_seconds() if pd.notna(laps_d1['Sector2Time'].min()) else None
                s2_d2 = laps_d2['Sector2Time'].min().total_seconds() if pd.notna(laps_d2['Sector2Time'].min()) else None

                s3_d1 = laps_d1['Sector3Time'].min().total_seconds() if pd.notna(laps_d1['Sector3Time'].min()) else None
                s3_d2 = laps_d2['Sector3Time'].min().total_seconds() if pd.notna(laps_d2['Sector3Time'].min()) else None

                sector_data = {
                    "Sector": ["Sector 1", "Sector 2", "Sector 3"],
                    f"{d1} Best (s)": [f"{s1_d1:.3f}" if s1_d1 else "-", f"{s2_d1:.3f}" if s2_d1 else "-", f"{s3_d1:.3f}" if s3_d1 else "-"],
                    f"{d2} Best (s)": [f"{s1_d2:.3f}" if s1_d2 else "-", f"{s2_d2:.3f}" if s2_d2 else "-", f"{s3_d2:.3f}" if s3_d2 else "-"],
                    "Delta Advantage": [
                        f"{d1} (-{abs(s1_d1 - s1_d2):.3f}s)" if s1_d1 and s1_d2 and s1_d1 < s1_d2 else f"{d2} (-{abs(s1_d2 - s1_d1):.3f}s)" if s1_d1 and s1_d2 else "-",
                        f"{d1} (-{abs(s2_d1 - s2_d2):.3f}s)" if s2_d1 and s2_d2 and s2_d1 < s2_d2 else f"{d2} (-{abs(s2_d2 - s2_d1):.3f}s)" if s2_d1 and s2_d2 else "-",
                        f"{d1} (-{abs(s3_d1 - s3_d2):.3f}s)" if s3_d1 and s3_d2 and s3_d1 < s3_d2 else f"{d2} (-{abs(s3_d2 - s3_d1):.3f}s)" if s3_d1 and s3_d2 else "-"
                    ]
                }
                st.dataframe(pd.DataFrame(sector_data), hide_index=True, use_container_width=True)

            # --- RACE PACE DISTRIBUTION (BOX PLOT) ---
            if not quick_1.empty and not quick_2.empty:
                st.markdown("#### Race Pace Consistency & Dispersion")
                st.caption("Distribution of representative racing laps (excluding pit-stops and safety car periods). Tighter spread = higher consistency.")

                quick_combined = pd.concat([quick_1, quick_2]).copy()
                quick_combined['LapTimeSeconds'] = quick_combined['LapTime'].dt.total_seconds()

                fig_box = px.box(
                    quick_combined,
                    x="Driver",
                    y="LapTimeSeconds",
                    color="Driver",
                    points="all",  # Show individual lap points beside the box
                    color_discrete_map={d1: "#FF1801", d2: "#00D2BE"},
                    title=f"{selected_team}: {d1} vs {d2} Pace Dispersion"
                )
                fig_box.update_layout(
                    template="plotly_dark",
                    yaxis_title="Lap Time (Seconds)",
                    xaxis_title="Driver",
                    height=500,
                    showlegend=False
                )
                st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.warning("Not enough teammate pairings detected for this session.")
    else:
        st.warning("Session results metadata is unavailable.")