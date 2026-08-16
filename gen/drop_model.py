import	torch
import	base.arch_model	as		model
from	sys				import	argv
nn_returns: model.feedforward_generation = model.feedforward_generation(device = "mps", input_size = 9).to("mps")
nn_returns.eval()
nn_returns.load_state_dict(torch.load(f'model/{argv[1]}_9_model_params.pth'))
torch.onnx.export(nn_returns, torch.rand((9, ), device = "mps"), f"debug/{argv[1]}_9_model.onnx")