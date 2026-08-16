import pandas as pd
from sys import argv
print(pd.read_parquet(f'test/signed_returns_{argv[2]}/{argv[1]}_returns_train.parquet'))
print(pd.read_parquet(f'test/signed_returns_{argv[2]}/{argv[1]}_returns_test.parquet'))
print(pd.read_parquet(f'test/unsigned_returns_{argv[2]}/{argv[1]}_returns_train.parquet'))
print(pd.read_parquet(f'test/unsigned_returns_{argv[2]}/{argv[1]}_returns_test.parquet'))