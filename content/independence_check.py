import	dcor
import	pandas		as		pd
import	numpy		as		np
from	sys			import	argv
price_df: pd.DataFrame = pd.read_parquet(f'../data/{argv[1]}_data.parquet')
print(f"Loaded {argv[1]}'s price dataframe")
vwap_series: pd.Series = np.log(price_df.vwap / price_df.vwap.shift(1)).dropna()
pos_diff = int(argv[2]) if len(argv) >= 3 else 10
arr: np.ndarray = pd.DataFrame({'ret1': vwap_series, 'ret2': vwap_series.shift(pos_diff)}).dropna().values
print("Now for the main part (signed)")
x = arr[:, 0]
y = arr[:, 1]
distance_corr = dcor.distance_correlation(x, y, method = 'avl')
p_value = dcor.independence.distance_correlation_t_test(x, y).pvalue
print(distance_corr, p_value)
print("Now for the main part (unsigned)")
x = np.abs(arr[:, 0])
y = np.abs(arr[:, 1])
distance_corr = dcor.distance_correlation(x, y, method = 'avl')
p_value = dcor.independence.distance_correlation_t_test(x, y).pvalue
print(distance_corr, p_value)