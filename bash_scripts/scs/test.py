import torch

x = torch.randn(10)
print(torch.__version__)
print(torch.version.cuda)
print(x)
print(x.cuda())
