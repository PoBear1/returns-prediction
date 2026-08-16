import	torch
import	numpy			as		np
import	pandas			as		pd
import	base.arch_model	as		model
from	sys				import	argv
# how do I import this model?
ret_nums = 9
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_vailable() else "cpu")
nn_returns: model.feedforward_generation = model.feedforward_generation(device = device, input_size = ret_nums).to(device)
nn_returns.eval()
nn_returns.load_state_dict(torch.load(f'model/{argv[1]}_{ret_nums}_model_params.pth'))
# ok so now I've imported this model, what do I do with it? 
# someone feeds me a tensor of prices, or maybe someone just feeds me a list of prices
# --input--
# I take the last 10, turn it into a returns series, get the numpy values, turn it into a torch tensor
# --process--
# I feed the returns into the model, and get mean M + uncertainty S out
# once I have this I then generate an independent t-distribution v = 2 variable N and then return price[-1] * np.exp(M + N * S)
def generate_price_prediction(prices: list[float]) -> float:
	last_prices: pd.Series = pd.Series(prices[-ret_nums - 1:])
	last_returns: torch.Tensor = torch.tensor((np.log(last_prices / last_prices.shift(1)).dropna() * 1e4).values)
	output: torch.Tensor = nn_returns(last_returns)
	mean: float = output[0].item() / 1e4
	uncertainty: float = np.sqrt(output[1].item()) / 1e4
	noise: float = np.random.standard_t(df = 2)
	return prices[-1] * np.exp(mean + uncertainty * noise)
	