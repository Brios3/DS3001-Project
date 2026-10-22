# Removes the train/validation/test labels from the Stage 4 dataset
 
import pandas as pd
from pathlib import Path
 

DATA_DIR = Path(r"C:\Users\bryso\PycharmProjects\PythonProject\Stat 2120\Raw Data")
INPUT_FILE = DATA_DIR / "merged_player_stage4_model_ready.csv"
 
if not INPUT_FILE.exists(): raise FileNotFoundError("File not found: " + str(INPUT_FILE))
 
df = pd.read_csv(INPUT_FILE)
df.columns = df.columns.str.strip()

if "SPLIT" in df.columns: df = df.drop(columns=["SPLIT"])
else: print("SPLIT column is not there")

df = df.sort_values(["SEASON", "NBA_ID"]).reset_index(drop=True)
df.to_csv(INPUT_FILE, index=False)
 
print("Stage 4 Player cleaning done")

