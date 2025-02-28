import torch, torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile
from model import ResNet, ResidualBlock
import os
loaded = torch.load("data.pth")
#model = torch.load("model.pth").to("cpu")
class_names = os.listdir("venv/animals")
class_names.sort()

num_classes = len(class_names)
model = ResNet(ResidualBlock, [3, 4, 6, 3], num_classes).cpu()
model.load_state_dict(loaded["model"])
#model = torchvision.models.mobilenet_v2(pretrained=True)
model.eval()
example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example)
traced_script_module_optimized = optimize_for_mobile(traced_script_module)
traced_script_module_optimized._save_for_lite_interpreter("model3.ptl")