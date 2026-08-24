import pandas as pd
from pathlib import Path
for file in list(Path(".").glob("*.parquet")):
	df = pd.read_parquet(file)
	df.to_parquet(file)
	print("Finished with", file)
	