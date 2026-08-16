import pandas as pd
from sys import argv
print(pd.read_parquet(f'data/{argv[1]}_data.parquet'))