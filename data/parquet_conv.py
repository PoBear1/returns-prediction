import pandas as pd
from pathlib import Path
for file in list(Path(".").glob("*.csv")):
	file = str(file)[:-4]
	df = pd.read_csv(file + ".csv")
	df.to_parquet(file + ".parquet", engine = "pyarrow", compression = "snappy")
