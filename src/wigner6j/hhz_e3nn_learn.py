import torch
from e3nn import o3

torch.manual_seed(0)

irreps_in1 = o3.Irreps("128x0e+128x1e")
irreps_in2 = o3.Irreps("128x0e+128x1e")
irreps_out = o3.Irreps("128x0e+128x1e")

tp = o3.FullyConnectedTensorProduct(
    irreps_in1, irreps_in2, irreps_out,
    shared_weights=False,
    internal_weights=False,
)

B = 4
x = torch.randn(B, irreps_in1.dim)
y = torch.randn(B, irreps_in2.dim)
w = torch.randn(B, tp.weight_numel)

out = tp(x, y, w)
print("in1 dim", irreps_in1.dim, "in2 dim", irreps_in2.dim, "out dim", irreps_out.dim)
print("weight_numel", tp.weight_numel)
print("out shape", out.shape)
# tp.visualize()  # 你本地可以打开看图
