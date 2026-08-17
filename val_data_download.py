import	pandas						as		pd
import	dotenv						as		env
from	sys							import	argv
from	zoneinfo					import	ZoneInfo
from	alpaca.data.timeframe		import	TimeFrame
from	alpaca.data.models.bars		import	Bar, BarSet
from	alpaca.trading.client		import	TradingClient
from	alpaca.data.requests		import	StockBarsRequest
from	alpaca.common.exceptions	import	APIError
from	datetime					import	datetime, timedelta
from	alpaca.data.historical		import	StockHistoricalDataClient

keys: dict[str, str] = env.dotenv_values()

data_client: StockHistoricalDataClient = StockHistoricalDataClient(
	   api_key = keys["API_PUBLIC_KEY"],
	secret_key = keys["API_SECRET_KEY"]
)
asset_checking_client: TradingClient = TradingClient(
	   api_key = keys["API_PUBLIC_KEY"],
	secret_key = keys["API_SECRET_KEY"],
		 paper = True
)
symbols: list[str] = []
for symbol in argv[1:]:
	try:
		asset = asset_checking_client.get_asset(symbol)
		print(f"{symbol} exists with status {asset.status}, tradeable: {asset.tradable}")
		symbols.append(symbol)
	except APIError as e:
		if e.status_code == 404:
			print(f"{symbol} does not exist")
		else:
			print(f"Weird API error: {e}")
			exit()
if len(symbols) == 0:
	exit()
timezone: ZoneInfo = ZoneInfo("America/New_York")
start_time: datetime = datetime(2020, 8, 1, 0, 0, 0, tzinfo = timezone)
end_time: datetime = datetime(2026, 8, 16, 0, 0, 0, tzinfo = timezone) 
data: dict[str, list[pd.DataFrame]] = {symbol: [] for symbol in symbols}

while start_time != end_time:
	for symbol in symbols:
		req_param: StockBarsRequest = StockBarsRequest(
			timeframe = TimeFrame.Minute,
			symbol_or_symbols = symbol,
			start = start_time,
			end = start_time + timedelta(days = 1)
		)
		data_list: BarSet = data_client.get_stock_bars(req_param)
		for bar in data_list.data:
			data[symbol].append(data_list.df.reset_index().drop(columns = ["symbol"]))
		print(f"Finished processing request for symbol {symbol} between days {start_time} and {start_time + timedelta(days = 1)}")
	start_time = start_time + timedelta(days = 1)
download_path = "data/{}_data.parquet"
for symbol in symbols:
	print("Downloading to \"" + (download_path.format(symbol)) + "\"...")
	final_price: pd.DataFrame = pd.concat(data[symbol], ignore_index = True)
	final_price.to_parquet(download_path.format(symbol), index = False)
	print("Downloaded to \"" + (download_path.format(symbol)) + "\".")