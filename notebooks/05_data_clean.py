# Removes the train/validation/test labels from the Stage 4 dataset
 
import pandas as pd
from pathlib import Path
 
# File location
DATA_DIR = Path(r"C:\Users\bryso\PycharmProjects\PythonProject\Stat 2120\Raw Data")
INPUT_FILE = DATA_DIR / "merged_player_stage4_model_ready.csv"
 
if not INPUT_FILE.exists(): raise FileNotFoundError("File not found: " + str(INPUT_FILE))
 
# Load data and strip stray spaces from column names
df = pd.read_csv(INPUT_FILE)
df.columns = df.columns.str.strip()
 
# Remove the split column if it's there
if "SPLIT" in df.columns: df = df.drop(columns=["SPLIT"])
else: print("SPLIT column is already absent.")
 
# Sort by season and player, then overwrite the file
df = df.sort_values(["SEASON", "NBA_ID"]).reset_index(drop=True)
df.to_csv(INPUT_FILE, index=False)
 
print("Stage 4 cleaning completed successfully!")
print("SPLIT column removed:", "SPLIT" not in df.columns)
print("Dataset shape:", df.shape)
print("Saved to:", INPUT_FILE)
