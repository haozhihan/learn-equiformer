import sys

try:
    import torch
    import e3nn
    from e3nn import o3
    from e3nn.o3 import spherical_harmonics, FullyConnectedTensorProduct
except ImportError as e:
    print(f"错误: 缺少必要的依赖包 - {e}")
    print("请先安装: pip install torch e3nn")
    sys.exit(1)

# 导入当前实现
from pathlib import Path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from src.wigner6j.tensor_product import (
        DepthWiseTensorProduct_reducesameorder,
        E2TensorProductArbitraryOrder,
    )
except ImportError as e:
    print(f"错误: 无法导入当前实现 - {e}")
    print("请确保在项目根目录运行此脚本")
    sys.exit(1)

def get_tp_e3nnsh_diff():
    # torch.set_printoptions(precision=8)
    Y = torch.randn(1, 3)

    self__square_tp = DepthWiseTensorProduct_reducesameorder(
        "1x1e",
        "1x1e",
        "1x2e",
        irrep_normalization="component",
        path_normalization="none",
    )
    self__square_tp.weight = torch.nn.Parameter(
        torch.ones(self__square_tp.weight.size()), requires_grad=False
    )
    Y_sq = self__square_tp(Y, Y)  # batch_size \times m time 5

    # self__tri_tp = DepthWiseTensorProduct_reducesameorder(
    #     "1x2e",
    #     "1x1e",
    #     "1x3e",
    #     irrep_normalization="component",
    #     path_normalization="none",
    # )
    # self__tri_tp.weight = torch.nn.Parameter(
    #     torch.ones(self__tri_tp.weight.size()), requires_grad=False
    # )
    # Y_tr = self__tri_tp(Y_sq, Y)  # batch_size \times m time 5

    # self__4_tp = DepthWiseTensorProduct_reducesameorder(
    #     "1x3e",
    #     "1x1e",
    #     "1x4e",
    #     irrep_normalization="component",
    #     path_normalization="none",
    # )
    # self__4_tp.weight = torch.nn.Parameter(
    #     torch.ones(self__4_tp.weight.size()), requires_grad=False
    # )
    # Y_4 = self__4_tp(Y_tr, Y)  # batch_size \times m time 5

    # self__5_tp = DepthWiseTensorProduct_reducesameorder(
    #     "1x4e",
    #     "1x1e",
    #     "1x5e",
    #     irrep_normalization="component",
    #     path_normalization="none",
    # )
    # self__5_tp.weight = torch.nn.Parameter(
    #     torch.ones(self__5_tp.weight.size()), requires_grad=False
    # )
    # Y_5 = self__5_tp(Y_4, Y)  # batch_size \times m time 5

    # self__6_tp = DepthWiseTensorProduct_reducesameorder(
    #     "1x5e",
    #     "1x1e",
    #     "1x6e",
    #     irrep_normalization="component",
    #     path_normalization="none",
    # )
    # self__6_tp.weight = torch.nn.Parameter(
    #     torch.ones(self__6_tp.weight.size()), requires_grad=False
    # )
    # Y_6 = self__6_tp(Y_5, Y)  # batch_size \times m time 5

    print(
        Y_sq
        / (e3nn.o3.spherical_harmonics(2, Y, normalize=False, normalization="integral"))
    )  # 1.29441716
    # print(
    #     Y_tr
    #     / (e3nn.o3.spherical_harmonics(3, Y, normalize=False, normalization="integral"))
    # )  # 0.84739512
    # print(
    #     Y_4
    #     / (e3nn.o3.spherical_harmonics(4, Y, normalize=False, normalization="integral"))
    # )  # 0.56493002
    # print(
    #     Y_5
    #     / (e3nn.o3.spherical_harmonics(5, Y, normalize=False, normalization="integral"))
    # )  # 0.38087577
    # print(
    #     Y_6
    #     / (e3nn.o3.spherical_harmonics(6, Y, normalize=False, normalization="integral"))
    # )  # 0.25875416


if __name__ == '__main__':
    get_tp_e3nnsh_diff()