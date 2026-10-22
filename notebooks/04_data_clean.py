
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(r"C:\Users\bryso\PycharmProjects\PythonProject\Stat 2120\Raw Data")

INPUT_FILE = DATA_DIR / "merged_player_stage3_modeling.csv"
OUTPUT_FILE = DATA_DIR / "player_final_data.csv"

identifiers = ["NBA_ID", "PLAYER_NAME", "SEASON", "POSITION"]

features = [
    "AGE", "MIN_PER_GAME", "GP_TOTAL",
    "PTS_PER12", "FGM_PER12", "FGA_PER12",
    "FG2M_PER12", "FG2A_PER12", "FG3M_PER12", "FG3A_PER12",
    "FTM_PER12", "FTA_PER12",
    "OREB_PER12", "DREB_PER12", "REB_PER12",
    "AST_PER12", "TOV_PER12", "STL_PER12", "BLK_PER12",
    "BLKA_PER12", "PF_PER12", "PFD_PER12",
    "FG_PCT", "FG2_PCT", "FG3_PCT", "FT_PCT", "EFG_PCT",
    "THREE_POINT_ATTEMPT_SHARE", "FREE_THROW_ATTEMPT_RATE",
    "AST_TO_TOV", "OREB_SHARE", "DD2_PER_GAME", "TD3_PER_GAME"
]

df = pd.read_csv(INPUT_FILE, low_memory=False)

required_columns = identifiers + features
missing = sorted(set(required_columns) - set(df.columns))

if missing:
    raise ValueError(f"Missing columns: {missing}")

if df[identifiers].isna().any().any():
    raise ValueError("Missing player identifiers or positions found.")

if df.duplicated(["NBA_ID", "SEASON"]).any():
    raise ValueError("Duplicate player-season records found.")

final_data = df[required_columns].copy()

numeric_data = final_data[features].apply(pd.to_numeric, errors="coerce")

invalid = (final_data[features].notna() & numeric_data.isna()).any()

if invalid.any():
    raise ValueError(f"Non-numeric values found in: {invalid[invalid].index.tolist()}")

final_data[features] = numeric_data

if np.isinf(final_data[features].to_numpy(dtype=float)).any():
    raise ValueError("Infinite predictor values found.")

final_data = final_data.sort_values(
    ["SEASON", "NBA_ID"]
).reset_index(drop=True)

final_data.to_csv(OUTPUT_FILE, index=False)

print("Stage 4 Player data cleaning done")
