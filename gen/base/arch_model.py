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
	def __init__(self, lmbd: float = 0.01, eta: float = 0.1, eps: float = 1e-7) -> None:
		super().__init__()
		self.lmbd = lmbd
		self.eps = eps
		self.eta = eta
	# outputs = (N, 2), targets = (N, )
	def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
		deviations: torch.Tensor = outputs[:, 0] - targets
		sq_dev: torch.Tensor = deviations * deviations
		inner_error: torch.Tensor = sq_dev / (2 * outputs[:, 1] + self.eps)
		log_error: torch.Tensor = 1.5 * torch.log(inner_error + 1 + self.eps) + 0.5 * torch.log(outputs[:, 1])
		var_reg: torch.Tensor = torch.abs(outputs[:, 1] + self.eps) * self.lmbd
		mean_reg: torch.Tensor = sq_dev * self.eta
		return (log_error + var_reg + mean_reg).mean()
class feedforward_generation(nn.Module):
	def __init__(self, device: str = "cpu", input_size: int = 9) -> None:
		super().__init__()
		self.dev = device
		self.internal_rep1 = 30
		self.internal_rep2 = 20
		self.internal_rep3 = 10
		self.project_up: nn.Linear = nn.Linear(in_features = input_size, out_features = self.internal_rep1, bias = True, device = self.dev)
		self.project_down_1: nn.Linear = nn.Linear(in_features = self.internal_rep1, out_features = self.internal_rep2, bias = True, device = self.dev)
		self.project_down_2: nn.Linear = nn.Linear(in_features = self.internal_rep2, out_features = self.internal_rep3, bias = True, device = self.dev)
		self.project_down_3: nn.Linear = nn.Linear(in_features = self.internal_rep3, out_features = 10, bias = True, device = self.dev)
		self.mlp_stack1: nn.Sequential = nn.Sequential(
			nn.Linear(in_features = self.internal_rep1, out_features = self.internal_rep1, bias = True, device = self.dev),
			nn.ReLU(),
			nn.Linear(in_features = self.internal_rep1, out_features = self.internal_rep1, bias = True, device = self.dev),
			nn.ReLU(),
			nn.Linear(in_features = self.internal_rep1, out_features = self.internal_rep1, bias = True, device = self.dev),
			nn.ReLU()
		)
		self.mlp_stack2: nn.Sequential = nn.Sequential(
			nn.Linear(in_features = self.internal_rep2, out_features = self.internal_rep2, bias = True, device = self.dev),
			nn.ReLU(),
			nn.Linear(in_features = self.internal_rep2, out_features = self.internal_rep2, bias = True, device = self.dev),
			nn.ReLU(),
			nn.Linear(in_features = self.internal_rep2, out_features = self.internal_rep2, bias = True, device = self.dev),
			nn.ReLU()
		)
		self.mlp_stack3: nn.Sequential = nn.Sequential(
			nn.Linear(in_features = self.internal_rep3, out_features = self.internal_rep3, bias = True, device = self.dev),
			nn.ReLU(),
			nn.Linear(in_features = self.internal_rep3, out_features = self.internal_rep3, bias = True, device = self.dev),
			nn.ReLU(),
			nn.Linear(in_features = self.internal_rep3, out_features = self.internal_rep3, bias = True, device = self.dev),
			nn.ReLU()
		)
		
		self.project_down: nn.Sequential = nn.Sequential(
			nn.Linear(in_features = self.internal_rep3, out_features = 10, bias = True, device = self.dev),
			nn.ReLU(),
			nn.Linear(in_features = 10, out_features = 2, bias = True, device = self.dev)
		)
	def forward(self, returns: torch.Tensor) -> torch.Tensor:
		project_1: torch.Tensor = self.project_up(returns)
		res1: torch.Tensor = self.mlp_stack1(project_1)
		raw_output1: torch.Tensor = res1 + project_1
		project_2: torch.Tensor = self.project_down_1(raw_output1)
		res2: torch.Tensor = self.mlp_stack2(project_2)
		raw_output2: torch.Tensor = res2 + project_2
		project_3: torch.Tensor = self.project_down_2(raw_output2)
		res3: torch.Tensor = self.mlp_stack3(project_3)
		raw_output3: torch.Tensor = res3 + project_3
		raw_output: torch.Tensor = self.project_down(raw_output3)
		output = torch.stack([raw_output[..., 0], raw_output[..., 1] * raw_output[..., 1]], dim = -1)
		return output
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
	def __init__(self, dataloader: tuple[DataLoader, DataLoader], lr: float = 0.001, momentum: float = 0.01, lmbd: float = 0.01, eta: float = 0.1, eps: float = 1e-7, in_size: int = 9, warmups: int = 4000, multi: float = 1.0, device: str = "cpu") -> None:
		self.model: nn.Module = (feedforward_generation)(device, in_size).to(device)
		self.loss_fn: nn.Module = (t_loss_function)(lmbd, eta, eps).to(device)
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
			pred: torch.Tensor = self.model(X)
			loss: torch.Tensor = self.loss_fn(pred, y)
			loss.backward()
			self.optimiser.step()
			self.optimiser.zero_grad()
			if (batch + 1) % 100 == 0:
				loss_history.append([loss.item(), (batch + 1) * len(X)])
		return [loss_history, size]
	def test(self) -> tuple[float, float, float]:
		num_batches: int = len(self.tests_dl)
		self.model.eval()
		test_loss: float = 0
		uncertainty_size: float = 0
		mean_size: float = 0
		print(avg_param_magnitude(self.model))
		with torch.no_grad():
			for X, y in self.tests_dl:
				X, y = X.to(self.device), y.to(self.device)
				pred: torch.Tensor = self.model(X)
				loss: torch.Tensor = self.loss_fn(pred, y)
				test_loss += loss.item()
				uncertainty_size += pred[:, 1].mean()
				mean_size += pred[:, 0].mean()
		return test_loss / num_batches, uncertainty_size / num_batches, mean_size / num_batches
	def drop_parameters(self, param_loc: str) -> None:
		torch.save(self.model.state_dict(), param_loc)
class returns_dataset(Dataset):
	def __init__(self, path: str, target_col: str = "pred_return"):
		df: pd.DataFrame = pd.read_parquet(path)
		feature_cols = [c for c in df.columns if c != target_col]
        # .values -> numpy, then to tensor once (not per __getitem__ call)
		self.X = torch.tensor(df[feature_cols].values, dtype = torch.float32)
		self.y = torch.tensor(df[target_col].values, dtype = torch.float32)
	def __len__(self) -> int:
		return self.X.shape[0]
	def __getitem__(self, idx: int):
		return self.X[idx], self.y[idx]