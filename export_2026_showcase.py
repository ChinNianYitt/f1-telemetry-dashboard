import os
import shutil
import tempfile
import fastf1
import pandas as pd

# Temporary directory for fast raw downloads
temp_cache = os.path.join(tempfile.gettempdir(), "f1_temp_cache")
os.makedirs(temp_cache, exist_ok=True)
fastf1.Cache.enable_cache(temp_cache)

out_dir = "showcase_data"
os.makedirs(out_dir, exist_ok=True)

year = 2026
schedule = fastf1.get_event_schedule(year)

# Filter for official championship rounds only
official_rounds = schedule[schedule['RoundNumber'] > 0].copy()

# Locate Hungary round number
hungary_events = official_rounds[official_rounds['EventName'].str.contains('Hungary|Hungarian', case=False, na=False)]
hungary_round = int(hungary_events['RoundNumber'].iloc[0]) if not hungary_events.empty else 13

# Filter from Hungary onward up to past events
now = pd.Timestamp.now(tz='UTC')
official_rounds['EventDateUTC'] = pd.to_datetime(official_rounds['EventDate'], utc=True)

target_events = official_rounds[
    (official_rounds['RoundNumber'] >= hungary_round) & 
    (official_rounds['EventDateUTC'] <= now)
]

print(f"Found {len(target_events)} completed rounds from Hungary onward for {year}.\n")

for _, event in target_events.iterrows():
    round_num = int(event['RoundNumber'])
    event_name = event['EventName']
    
    # Generate clean file slug: e.g. "2026_13_hungarian"
    clean_name = event_name.lower().replace("grand prix", "").strip().replace(" ", "_")
    slug = f"{year}_{round_num}_{clean_name}"
    
    laps_file = os.path.join(out_dir, f"{slug}_laps.parquet")
    results_file = os.path.join(out_dir, f"{slug}_results.parquet")
    tel_file = os.path.join(out_dir, f"{slug}_telemetry.parquet")

    # Skip if all three files already exist
    if os.path.exists(laps_file) and os.path.exists(results_file) and os.path.exists(tel_file):
        print(f"⏭️ Skipping Round {round_num}: {event_name} (already exported).")
        continue

    print(f"\n==========================================")
    print(f"Downloading Round {round_num}: {event_name}...")
    print(f"==========================================")
    
    try:
        session = fastf1.get_session(year, round_num, "R")
        session.load(telemetry=True, laps=True, weather=False)
        
        # 1. Export Laps
        session.laps.to_parquet(laps_file, index=False)
        print(f" Saved Laps: {laps_file}")
        
        # 2. Export Results Metadata
        res_cols = ['Abbreviation', 'FullName', 'TeamName', 'Position', 'ClassifiedPosition', 'GridPosition', 'Status', 'Points']
        valid_cols = [c for c in res_cols if c in session.results.columns]
        results_df = session.results[valid_cols]
        results_df.to_parquet(results_file, index=False)
        print(f" Saved Results: {results_file}")
        
        # 3. Export Driver Fastest Lap Telemetry with GPS (X, Y)
        tel_frames = []
        for drv in session.drivers:
            try:
                drv_laps = session.laps.pick_driver(drv)
                fastest = drv_laps.pick_fastest()
                if fastest is not None:
                    tel = fastest.get_telemetry()
                    
                    abbr_matches = session.results.loc[
                        (session.results['DriverNumber'] == str(drv)) | 
                        (session.results['Abbreviation'] == str(drv)), 
                        'Abbreviation'
                    ]
                    drv_label = abbr_matches.values[0] if not abbr_matches.empty else str(drv)
                    tel['Driver'] = drv_label
                    
                    channels = ['Distance', 'Speed', 'Throttle', 'Brake', 'nGear', 'Time', 'Driver', 'X', 'Y', 'Z']
                    valid_ch = [c for c in channels if c in tel.columns]
                    tel_frames.append(tel[valid_ch])
            except Exception:
                continue
                
        if tel_frames:
            telemetry_df = pd.concat(tel_frames, ignore_index=True)
            telemetry_df.to_parquet(tel_file, index=False)
            print(f" Saved Telemetry ({len(telemetry_df)} rows) with GPS channels.")

    except Exception as e:
        print(f"⚠️ Could not export {event_name}: {e}")

# Clear the temporary download cache to reclaim space immediately
shutil.rmtree(temp_cache, ignore_errors=True)
print("\nExport completed! Clean showcase files saved in 'showcase_data/'.")