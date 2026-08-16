import	torch
import	torch.nn.functional	as		F
import	pandas				as		pd
import	torch.nn			as		nn
import	torch.optim			as		optim
from	torch.utils.data	import	DataLoader, Dataset

import	base.arch_model		as		model

from	sys					import	argv

batch_size: int = 50
epochs: int = 1000
num_rets: int = int(argv[2]) if len(argv) >= 3 else 9 
signed = argv[3] == 'Y' if len(argv) >= 4 else True

symbol_name: str = argv[1]

returns_data: tuple[model.returns_dataset, model.returns_dataset] = (
	model.returns_dataset(f'test/{'un' if not signed else ''}signed_returns_{num_rets}/{symbol_name}_returns_train.parquet'),
	model.returns_dataset(f'test/{'un' if not signed else ''}signed_returns_{num_rets}/{symbol_name}_returns_test.parquet')
)

returns_dataloaders: tuple[DataLoader, DataLoader] = (
	DataLoader(returns_data[0], batch_size = batch_size),
	DataLoader(returns_data[1], batch_size = batch_size)
)

lr: float = 10

model_trainer: model.generator_trainer = model.generator_trainer(returns_dataloaders, lr, momentum = 0.01, in_size = num_rets, lmbd = 0.1, warmups = 40, multi = 1000, device = "mps")
for t in range(epochs):
	print(f"Epoch {t + 1}\n-------------------------------")
	history, size = model_trainer.train()
	for loss, current in history:
		print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
	test_loss, uncertainty_size, mean_size = model_trainer.test()
	print(f"Test Error: \n Avg loss: {test_loss:>8f}, Average uncertainty: {uncertainty_size:>8f}, Average mean: {mean_size:>8f} \n")
model_trainer.drop_parameters(f'model/{symbol_name}_{num_rets}_model_params.pth')