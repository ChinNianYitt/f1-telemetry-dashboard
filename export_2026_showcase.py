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

# Target the two 2026 rounds
targets = [
    (2026, "Azerbaijan", "R", "2026_azerbaijan"),
    (2026, "Bahrain", "R", "2026_bahrain")
]

for year, gp, session_type, slug in targets:
    print(f"\n==========================================")
    print(f"Downloading {year} {gp} Grand Prix...")
    print(f"==========================================")
    
    session = fastf1.get_session(year, gp, session_type)
    # Load all telemetry streams including car position GPS data
    session.load(telemetry=True, laps=True, weather=False)
    
    # 1. Export Laps
    laps_file = os.path.join(out_dir, f"{slug}_laps.parquet")
    session.laps.to_parquet(laps_file, index=False)
    print(f" Saved Laps: {laps_file}")
    
    # 2. Export Results Metadata
    res_cols = ['Abbreviation', 'FullName', 'TeamName', 'Position', 'ClassifiedPosition', 'GridPosition', 'Status', 'Points']
    valid_cols = [c for c in res_cols if c in session.results.columns]
    results_df = session.results[valid_cols]
    results_file = os.path.join(out_dir, f"{slug}_results.parquet")
    results_df.to_parquet(results_file, index=False)
    print(f" Saved Results: {results_file}")
    
    # 3. Export Driver Fastest Lap Telemetry with GPS (X, Y)
    tel_frames = []
    for drv in session.drivers:
        try:
            drv_laps = session.laps.pick_driver(drv)
            fastest = drv_laps.pick_fastest()
            if fastest is not None:
                # Merge car data (Speed, Throttle, Brake, Gear) with GPS position data (X, Y)
                tel = fastest.get_telemetry()
                
                # Retrieve the 3-letter abbreviation (e.g. 'VER') matching app.py dropdowns
                abbr_matches = session.results.loc[
                    (session.results['DriverNumber'] == str(drv)) | 
                    (session.results['Abbreviation'] == str(drv)), 
                    'Abbreviation'
                ]
                drv_label = abbr_matches.values[0] if not abbr_matches.empty else str(drv)
                tel['Driver'] = drv_label
                
                # Keep required channels including GPS coordinates
                channels = ['Distance', 'Speed', 'Throttle', 'Brake', 'nGear', 'Time', 'Driver', 'X', 'Y', 'Z']
                valid_ch = [c for c in channels if c in tel.columns]
                tel_frames.append(tel[valid_ch])
        except Exception as e:
            print(f"Driver {drv} extraction note: {e}")
            continue
            
    if tel_frames:
        telemetry_df = pd.concat(tel_frames, ignore_index=True)
        tel_file = os.path.join(out_dir, f"{slug}_telemetry.parquet")
        telemetry_df.to_parquet(tel_file, index=False)
        print(f" Saved Telemetry ({len(telemetry_df)} rows) with channels: {list(telemetry_df.columns)}")

# Clear the temporary download cache to reclaim space immediately
shutil.rmtree(temp_cache, ignore_errors=True)
print("\nExport completed! Clean files with X/Y coordinates saved in 'showcase_data/'.")