import pandas as pd
from pathlib import Path
for file in list(Path(".").glob("*.parquet")):
	df = pd.read_parquet(file)
	df.to_parquet(file, compression = 'zstd', compression_level = 22)
	print("Finished with", file)
	