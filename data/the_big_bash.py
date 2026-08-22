import pandas as pd
from pathlib import Path
for file in list(Path(".").glob("*.parquet")):
	df = pd.read_parquet(file)
	df['datetime'] = pd.to_datetime(df['timestamp'])
	df['timestamp'] = df['datetime'].astype('int64') // (6 * 10 ** 7)
	cols = ['timestamp', 'datetime'] + [c for c in df.columns if c not in ['timestamp', 'datetime']]
	df = df[cols]
	df = df.set_index("timestamp")
	df.to_parquet(file)
	print("Finished with", file)
	