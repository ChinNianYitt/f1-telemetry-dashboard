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

# Rounds from Hungary onward (Official 2026 Schedule)
targets = [
    (2026, 11, "Hungarian Grand Prix", "2026_11_hungary"),
    (2026, 12, "Dutch Grand Prix", "2026_12_dutch"),
    (2026, 13, "Italian Grand Prix", "2026_13_monza"),
    (2026, 14, "Spanish Grand Prix", "2026_14_spain"),
    (2026, 15, "Azerbaijan Grand Prix", "2026_15_azerbaijan"),
    (2026, 16, "Bahrain Grand Prix", "2026_16_bahrain"),
]

for year, round_num, gp_name, slug in targets:
    laps_file = os.path.join(out_dir, f"{slug}_laps.parquet")
    results_file = os.path.join(out_dir, f"{slug}_results.parquet")
    tel_file = os.path.join(out_dir, f"{slug}_telemetry.parquet")

    if os.path.exists(laps_file) and os.path.exists(results_file) and os.path.exists(tel_file):
        print(f"⏭️ Skipping Round {round_num}: {gp_name} (already exists).")
        continue

    print(f"\n==========================================")
    print(f"Downloading Round {round_num}: {gp_name}...")
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
            print(f" Saved Telemetry: {tel_file}")

    except Exception as e:
        print(f"❌ Failed to download {gp_name}: {e}")

shutil.rmtree(temp_cache, ignore_errors=True)
print("\nProcess finished.")