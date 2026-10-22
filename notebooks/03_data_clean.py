
# Stage 3: Validates NBA player statistics, flags unreliable
# observations, and prepares a sample for position classification.

import pandas as pd
import numpy as np
from pathlib import Path

Filepath = Path(r"C:\Users\bryso\PycharmProjects\PythonProject\Stat 2120\Raw Data")

Input = Filepath / "merged_player_stage2_cleaned.csv"
Output = Filepath / "merged_player_stage3_modeling.csv"
Output_audit = Filepath / "merged_player_stage3_audited.csv"

df = pd.read_csv(Input)

if df.duplicated(["NBA_ID", "SEASON"]).any():
    raise ValueError("Duplicate player-season observations detected.")

if df["POSITION"].isna().any():
    raise ValueError("Missing player positions detected.")

stat_columns = [col for col in df.columns
    if col.endswith("_PER12")]

if (df[stat_columns] < 0).any().any():
    raise ValueError("Negative player statistics detected.")

if np.isinf(df[stat_columns].to_numpy()).any():
    raise ValueError("Infinite player statistics detected.")

percentage_columns = ["FG_PCT","FG2_PCT","FG3_PCT","FT_PCT","EFG_PCT","THREE_POINT_ATTEMPT_SHARE","OREB_SHARE"]

for col in percentage_columns:
    valid = df[col].dropna().between(0, 1)

    if not valid.all():
        raise ValueError(f"Invalid percentage in {col}")

shot_pairs = [("FGM_PER12", "FGA_PER12"), ("FG3M_PER12", "FG3A_PER12"),("FG2M_PER12", "FG2A_PER12"), ("FTM_PER12", "FTA_PER12")]

df["SHOT_INCONSISTENCY_FLAG"] = False

for made, attempted in shot_pairs:
    df["SHOT_INCONSISTENCY_FLAG"] |= (
        df[made] > df[attempted] + 0.01
    ).fillna(False)

df["REBOUND_DISCREPANCY_FLAG"] = ((df["OREB_PER12"] + df["DREB_PER12"] - df["REB_PER12"]).abs() > 0.20)
df["LOW_MINUTES_SAMPLE"] = (df["MIN_TOTAL_APPROX"] < 120)

reference = df[df["MIN_TOTAL_APPROX"] >= 120]

df["EXTREME_PER12_FLAG"] = False

for col in stat_columns:

    upper = reference[col].quantile(0.999)

    df["EXTREME_PER12_FLAG"] |= (df[col] > upper).fillna(False)

df["MISSING_STAT_COUNT"] = (df[stat_columns + percentage_columns].loc[:, lambda x: ~x.columns.duplicated()].isna().sum(axis=1))

df.to_csv(Output_audit, index=False)

model_df = df[~df["LOW_MINUTES_SAMPLE"]].copy()

original_columns = pd.read_csv(Input, nrows=0).columns

model_df = model_df[original_columns]

model_df.to_csv(Output, index=False)

print("Stage 3 Player data cleaning done")
