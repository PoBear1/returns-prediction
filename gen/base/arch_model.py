import	torch
import	torch.nn.functional	as		F
import	pandas				as		pd
import	numpy				as		np
import	torch.nn			as		nn
import	torch.optim			as		optim
from	torch.utils.data	import	DataLoader, Dataset
def avg_param_magnitude(model):
    all_params = torch.cat([p.detach().flatten() for p in model.parameters()])
    return all_params.abs().mean().item()
# so currently our thing is generating µ, s for T(v = 2). 
# if we wanna keep it going then we wanna minimise -log(p((y - µ) / s)) = -log((1 + (y - µ)^2 / 2s^2) ^ -3/2) = 3/2 log(1 + (y - µ)^2 / 2s^2)
# say now that the model realises it can game the system by doing s -> oo, we thus need to regularise s as well. 
# i propose adding +lambda |s| to the loss function. 
# why?
# idk why lol
# here's the somewhat weird reason why I would.
# consider p(y|m(x), s(x)) ~ t(y|m(x), s(x))e^{-s(x)}
# so the larger the uncertainty, the less likely anything is anywhere. 
# this penalty increases much faster as s increases
# hopefully now the robot realises that s shouldn't be too big.
class t_loss_function(nn.Module):
	def __init__(self, lmbd: float = 0.01, sigma: float = 0.01, eta: float = 0.1) -> None:
		super().__init__()
		self.lmbd = lmbd
		self.sigma = sigma
		self.eta = eta
		self.nu = 2
	# mean_outputs = (N, num_assets), cov_outputs = (N, num_assets, num_assets), targets = (N, num_assets), output = (N, )
	def forward(self, mean_output: torch.Tensor, cov_output: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
		# auxiliary stuff
		num_assets: int = mean_output.shape[-1]
		offdiag_mask: torch.Tensor = ~torch.eye(num_assets, dtype = torch.bool, device = cov_output.device)
		cholesky_cov: torch.Tensor = torch.linalg.cholesky(cov_output)
		chol_cov_inv: torch.Tensor = torch.linalg.solve_triangular(cholesky_cov, torch.eye(num_assets, device = cov_output.device).expand_as(cholesky_cov), upper = False)
		cov_prec: torch.Tensor = chol_cov_inv.mT @ chol_cov_inv
		# (x - µ) and L(x - µ)
		deviations: torch.Tensor = mean_output - targets
		trans_deviations: torch.Tensor = torch.linalg.solve_triangular(cholesky_cov, deviations.unsqueeze(-1), upper = False)
		inner_error: torch.Tensor = (trans_deviations ** 2).sum(dim = (-1, -2))
		logdet_sqrt: torch.Tensor = torch.diagonal(cholesky_cov, dim1 = -2, dim2 = -1).log().sum(dim = -1)
		log_error: torch.Tensor = 0.5 * (num_assets + self.nu) * torch.log1p(inner_error / self.nu) + logdet_sqrt
		var_reg: torch.Tensor = self.lmbd * (torch.diagonal(cov_output, dim1 = -2, dim2 = -1) ** 2).sum(dim = -1)
		covar_reg: torch.Tensor = self.sigma * torch.abs(cov_prec[..., offdiag_mask]).sum(dim = -1)
		mean_reg: torch.Tensor = self.eta * (deviations.abs().sum(dim = -1))
		return (log_error + var_reg + covar_reg + mean_reg).mean()
class feedforward_block(nn.Module):
	def __init__(self, res_block: nn.Sequential, proj_block: nn.Linear, norm_block: nn.LayerNorm) -> None:
		self.res_block = res_block
		self.proj_block = proj_block
		self.norm_block = norm_block
	def forward(self, input: torch.Tensor) -> torch.Tensor:
		return self.norm_block(self.proj_block(input + self.res_block(input)))
class feedforward_generation(nn.Module):
	def __init__(self, device: str = "cpu", num_rets: int = 9, num_assets: int = 1, factors: int = 1) -> None:
		super().__init__()
		self.dev = device
		self.eps = 1e-7
		self.num_assets = num_assets
		self.factors = factors
		self.internal_rep_in: list[int] = [50 * num_assets, 40 * num_assets, 30 * num_assets, 20 * num_assets]
		self.internal_rep_out: list[int] = self.internal_rep_in[1:] + [10 * num_assets]
		self.input_size: int = num_assets * num_rets
		self.correction: torch.Tensor = torch.eye(self.num_assets, device = self.dev) * self.eps
		# start of the network in projection
		self.project_up: nn.Linear = nn.Linear(in_features = self.input_size, out_features = self.internal_rep_in[0], bias = False, device = self.dev)
		# the main body of the network
		self.mlp_stacks: nn.ModuleList[nn.ModuleList[nn.Sequential, nn.Linear, nn.LayerNorm]] = nn.ModuleList([
			feedforward_block(
				nn.Sequential(
					nn.Linear(in_features = in_rep, out_features = in_rep, bias = True, device = self.dev),
					nn.ReLU(),
					nn.Linear(in_features = in_rep, out_features = in_rep, bias = True, device = self.dev),
					nn.ReLU(),
					nn.Linear(in_features = in_rep, out_features = in_rep, bias = True, device = self.dev),
					nn.ReLU(),
					nn.Linear(in_features = in_rep, out_features = in_rep, bias = True, device = self.dev),
					nn.ReLU(),
					nn.Linear(in_features = in_rep, out_features = in_rep, bias = True, device = self.dev),
					nn.ReLU()
				), nn.Linear(in_features = in_rep, out_features = out_rep, bias = False, device = self.dev), nn.LayerNorm(normalized_shape = (out_rep,), device = self.dev)
			)
			for in_rep, out_rep in zip(self.internal_rep_in, self.internal_rep_out)
		])

		# mean head
		self.mean_head_rep_in: list[int] = [self.internal_rep_out[-1]] + [8 * num_assets, 4 * num_assets, 2 * num_assets]
		self.mean_head_rep_out: list[int] = self.mean_head_rep_in[1:] + [num_assets]
		self.mean_head_stacks: nn.ModuleList[nn.ModuleList[nn.Sequential, nn.Linear, nn.LayerNorm]] = nn.ModuleList([
			feedforward_block(
				nn.Sequential(
					nn.Linear(in_features = in_rep, out_features = in_rep, bias = True, device = self.dev),
					nn.ReLU(),
					nn.Linear(in_features = in_rep, out_features = in_rep, bias = True, device = self.dev),
					nn.ReLU(),
					nn.Linear(in_features = in_rep, out_features = in_rep, bias = True, device = self.dev),
					nn.ReLU(),
					nn.Linear(in_features = in_rep, out_features = in_rep, bias = True, device = self.dev),
					nn.ReLU(),
					nn.Linear(in_features = in_rep, out_features = in_rep, bias = True, device = self.dev),
					nn.ReLU()
				), nn.Linear(in_features = in_rep, out_features = out_rep, bias = False, device = self.dev), nn.LayerNorm(normalized_shape = (out_rep,), device = self.dev)
			)
			for in_rep, out_rep in zip(self.mean_head_rep_in, self.mean_head_rep_out)
		])
		self.mean_head_final_project: nn.Linear = nn.Linear(in_features = self.mean_head_rep_out[-1], out_features = num_assets, bias = True, device = self.dev)

		# covariance head
		self.cov_head_rep_in: list[int] = [self.internal_rep_out[-1]] + [8 * num_assets * factors, 4 * num_assets * factors, 2 * num_assets * factors]
		self.cov_head_rep_out: list[int] = self.cov_head_rep_in[1:] + [num_assets * factors]
		self.cov_head_stacks: nn.ModuleList[tuple[nn.Sequential, nn.Linear, nn.LayerNorm]] = nn.ModuleList([
			feedforward_block(
				nn.Sequential(
					nn.Linear(in_features = in_rep, out_features = in_rep, bias = True, device = self.dev),
					nn.ReLU(),
					nn.Linear(in_features = in_rep, out_features = in_rep, bias = True, device = self.dev),
					nn.ReLU(),
					nn.Linear(in_features = in_rep, out_features = in_rep, bias = True, device = self.dev),
					nn.ReLU(),
					nn.Linear(in_features = in_rep, out_features = in_rep, bias = True, device = self.dev),
					nn.ReLU(),
					nn.Linear(in_features = in_rep, out_features = in_rep, bias = True, device = self.dev),
					nn.ReLU()
				), nn.Linear(in_features = in_rep, out_features = out_rep, bias = False, device = self.dev), nn.LayerNorm(normalized_shape = (out_rep,), device = self.dev)
			)
			for in_rep, out_rep in zip(self.cov_head_rep_in, self.cov_head_rep_out)
		])
		self.cov_head_final_project: nn.Linear = nn.Linear(in_features = self.cov_head_rep_out[-1], bias = False, out_features = num_assets * factors, device = self.dev)

	def forward(self, returns: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		# main body
		body_output: torch.Tensor = self.project_up(returns.reshape((-1, self.input_size)))
		for block in self.mlp_stacks:
			body_output = block(body_output)

		# mean head
		head_output: torch.Tensor = body_output
		for block in self.mean_head_stacks:
			head_output = block(head_output)
		mean_output: torch.Tensor = self.mean_head_final_project(head_output)

		# cov head
		cov_output: torch.Tensor = body_output[-1]
		for block in self.cov_head_stacks:
			cov_output = block(cov_output)
		raw_cov_output: torch.Tensor = self.cov_head_final_project(cov_output)
		factor_cov_output: torch.Tensor = raw_cov_output.reshape(*raw_cov_output.shape[:-1], self.factors, self.num_assets)
		cov_output: torch.Tensor = factor_cov_output.mT @ factor_cov_output
		cov_output = cov_output + self.correction
		return mean_output, cov_output
class exp_lr_scheduler(optim.lr_scheduler._LRScheduler):
	def __init__(self, optimiser: optim.Optimizer, warmup_steps: int, multiplier: float, min_lr: float = 0.0) -> None:
		self.warmup: float = float(warmup_steps * warmup_steps * warmup_steps)
		self.multi: float = multiplier
		self.min_lr: float = min_lr
		super().__init__(optimiser)
	def get_lr(self):
		step: float = float(self.last_epoch) + 1e-7
		scale: float = np.maximum(np.minimum(1 / np.sqrt(step), step / np.sqrt(self.warmup)) * self.multi, self.min_lr)
		return [base_lr * scale for base_lr in self.base_lrs]
class generator_trainer:
	def __init__(self, dataloader: tuple[DataLoader, DataLoader], lr: float = 0.001, momentum: float = 0.01, lmbd: float = 0.01, eta: float = 0.1, in_size: int = 9, warmups: int = 4000, multi: float = 1.0, device: str = "cpu") -> None:
		self.model: nn.Module = (feedforward_generation)(device, in_size).to(device)
		self.loss_fn: nn.Module = (t_loss_function)(lmbd, eta).to(device)
		self.optimiser: optim.Optimizer = optim.SGD(self.model.parameters(), lr = lr, momentum = momentum, weight_decay = 0.0)
		# self.optimiser: optim.Optimizer = optim.Adam(self.model.parameters(), lr = lr, beta = (0.9, 0.98), eps = 1e-9, weight_decay = 0.0)
		self.scheduler: optim.lr_scheduler._LRScheduler = exp_lr_scheduler(self.optimiser, warmups, multi, min_lr = 0.0)
		self.train_dl: DataLoader = dataloader[0]
		self.tests_dl: DataLoader = dataloader[1]
		self.device: str = device
	def train(self) -> None:
		size: int = len(self.train_dl.dataset)
		loss_history: list[tuple[float, int]] = []
		self.model.train()
		for batch, (X, y) in enumerate(self.train_dl):
			X, y = X.to(self.device), y.to(self.device)
			pred_mean, pred_cov = self.model(X)
			loss: torch.Tensor = self.loss_fn(pred_mean, pred_cov, y)
			loss.backward()
			self.optimiser.step()
			self.optimiser.zero_grad()
			if (batch + 1) % 100 == 0:
				loss_history.append([loss.item(), (batch + 1) * len(X)])
		return [loss_history, size]
	def test(self) -> tuple[float, torch.Tensor, torch.Tensor]:
		num_batches: int = len(self.tests_dl)
		self.model.eval()
		test_loss: float = 0
		uncertainty_size: torch.Tensor = 0
		mean_size: torch.Tensor = 0
		print(avg_param_magnitude(self.model))
		with torch.no_grad():
			for X, y in self.tests_dl:
				X, y = X.to(self.device), y.to(self.device)
				pred_mean, pred_cov = self.model(X)
				loss: torch.Tensor = self.loss_fn(pred_mean, pred_cov, y)
				test_loss += loss.item()
				uncertainty_size += pred_cov.mean(dim = 0)
				mean_size += pred_mean.mean(dim = 0)
		return test_loss / num_batches, uncertainty_size / num_batches, mean_size / num_batches
	def drop_parameters(self, param_loc: str) -> None:
		torch.save(self.model.state_dict(), param_loc)
class returns_dataset(Dataset):
	def __init__(self, paths: list[str], target_col: str = ["pred_return"]):
		df_list: list[pd.DataFrame] = [pd.read_parquet(path) for path in paths]
		feature_cols_dict = [[c for c in df.columns if c not in target_col] for df in df_list]
        # .values -> numpy, then to tensor once (not per __getitem__ call)
		self.X = torch.stack([torch.tensor(df[feature_cols].values, dtype = torch.float32) for df, feature_cols in zip(df_list, feature_cols_dict)], dim = -2)
		self.y = torch.stack([torch.tensor(df[target_col].values.squeeze(-1), dtype = torch.float32) for df in df_list], dim = -1)
	def __len__(self) -> int:
		return self.X.shape[0]
	def __getitem__(self, idx: int):
		return self.X[idx], self.y[idx]