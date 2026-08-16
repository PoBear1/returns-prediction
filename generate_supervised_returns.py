import	numpy		as		np
import	pandas		as		pd
import	random		as		rand
from	sys			import	argv
from	subprocess	import	run
deterministic = False
rng = np.random.default_rng(seed = 42) if deterministic else np.random.default_rng()
price_data: pd.DataFrame = pd.read_parquet('data/' + argv[1] + '_data.parquet')
returns: pd.Series = np.log(price_data.vwap.shift(1) / price_data.vwap).dropna() * 1e4
num_rets = int(argv[2]) if len(argv) >= 3 else 9
tabular_returns_data_train = {f'returns_{i}': [] for i in range(num_rets)} | {'pred_return': []}
tabular_returns_data_test = {f'returns_{i}': [] for i in range(num_rets)} | {'pred_return': []}
indices = set()
num_samples: int = 60000
def idx(x): return x
signed = idx if argv[3] == 'Y' or len(argv) < 4 else np.abs
while len(indices) != 2 * num_samples:
	indices.add(rng.integers(len(returns) - num_rets))
	print(f"{argv[1]} has {len(indices)} elements now")
indices_train = set(rand.sample(list(indices), num_samples))
indices_test = indices ^ indices_train
num = 0
for i in indices_train:
	for j in range(num_rets):
		tabular_returns_data_train[f'returns_{j}'].append(signed(returns.loc[i + j]))
	tabular_returns_data_train[f'pred_return'].append(signed(returns.loc[i + num_rets]))
	num += 1
	print(f"{argv[1]} has {num} training rows now!")
num = 0
for i in indices_test:
	for j in range(num_rets):
		tabular_returns_data_test[f'returns_{j}'].append(signed(returns.loc[i + j]))
	tabular_returns_data_test[f'pred_return'].append(signed(returns.loc[i + num_rets]))
	num += 1
	print(f"{argv[1]} has {num} testing rows now!")

pd.DataFrame(tabular_returns_data_train).to_parquet(f'gen/test/{'signed' if signed == idx else 'unsigned'}_returns_{num_rets}/' + argv[1] + '_returns_train.parquet')
pd.DataFrame(tabular_returns_data_test).to_parquet(f'gen/test/{'signed' if signed == idx else 'unsigned'}_returns_{num_rets}/' + argv[1] + '_returns_test.parquet')