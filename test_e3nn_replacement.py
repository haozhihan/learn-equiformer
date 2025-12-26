"""
测试文件：对比当前 tensor_product_tp_component_1 实现与 e3nn TensorProduct

这个文件用于验证是否可以用 e3nn 的 TensorProduct 替换当前的实现。
不会修改原始代码，只进行对比测试。

使用方法:
    python test_e3nn_replacement.py

需要安装的依赖:
    - torch
    - e3nn
"""

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


def create_e3nn_tensor_product(irreps_in, irreps_out, order, learnable_weight=True, connection_mode="uvw"):
    """
    创建 e3nn 的 TensorProduct，尝试匹配当前实现的参数
    
    Args:
        irreps_in: 输入不可约表示
        irreps_out: 输出不可约表示
        order: 球谐函数阶数
        learnable_weight: 是否可学习权重
        connection_mode: 连接模式
        
    Returns:
        e3nn.o3.TensorProduct 实例
    """
    irreps_in = o3.Irreps(irreps_in)
    irreps_out = o3.Irreps(irreps_out)
    irreps_edge = o3.Irreps(f"1x{order}e")
    
    # 使用 FullyConnectedTensorProduct 自动生成 instructions，避免直接
    # 调用 TensorProduct 需要手写 instructions 的报错
    # 注意：当 internal_weights=True 时，shared_weights 也必须为 True
    tp = FullyConnectedTensorProduct(
        irreps_in1=irreps_in,
        irreps_in2=irreps_edge,
        irreps_out=irreps_out,
        internal_weights=learnable_weight,
        shared_weights=learnable_weight,  # 当 internal_weights=True 时，shared_weights 必须为 True
        irrep_normalization="component",  # 尝试与当前实现保持一致
        path_normalization="none",
    )
    
    return tp


def test_tensor_product_replacement():
    """测试替换的可行性"""
    
    print("=" * 80)
    print("测试 tensor_product_tp_component_1 与 e3nn TensorProduct 的对比")
    print("=" * 80)
    
    # 测试参数
    head, hidden = 2, 4
    f_N1, f_N2 = 4, 4
    order = 1
    
    # 创建测试数据
    torch.manual_seed(42)  # 固定随机种子以便复现
    
    alpha_ij = torch.randn(f_N1, f_N2, head)
    h = torch.randn(f_N1, (order + 1) ** 2, head * hidden)
    exp_h = torch.randn(f_N2, (order + 1) ** 2, head * hidden)
    pos = torch.randn(f_N1, 3)
    exp_pos = torch.randn(f_N2, 3)
    
    # 创建 irreps
    irreps_in = "+".join([
        f"{head*hidden}x0e",
        f"{head*hidden}x1e",
        # f"{head*hidden}x2e",
    ])
    irreps_out = irreps_in
    
    print(f"\n输入 irreps: {irreps_in}")
    print(f"输出 irreps: {irreps_out}")
    print(f"球谐函数阶数: {order}")
    
    # ========== 当前实现 ==========
    print("\n" + "=" * 80)
    print("1. 当前实现 (DepthWiseTensorProduct_reducesameorder)")
    print("=" * 80)
    
    current_tp = DepthWiseTensorProduct_reducesameorder(
        irreps_in,
        f"1x{order}e",
        irreps_out,
        irrep_normalization="component",
        path_normalization="none",
        learnable_weight=True,
        connection_mode="uvw",
    )
    
    # 准备输入数据（模拟 vanilla_forward 中的处理）
    delta_pos = pos.unsqueeze(dim=1) - exp_pos.unsqueeze(dim=0)
    coeffs = E2TensorProductArbitraryOrder.get_coeffs()
    delta_pos_order_l = (
        spherical_harmonics(
            order, delta_pos, normalize=False, normalization="integral"
        )
        * coeffs[order]
    )
    delta_pos_order_l = delta_pos_order_l.unsqueeze(dim=-1)
    
    h_new = exp_h.reshape(f_N2, -1, head, (head * hidden) // head)
    h_new = torch.einsum("bjh, johk -> bjohk", alpha_ij, h_new)
    h_new = h_new.reshape(f_N1, f_N2, -1, head * hidden)
    
    print(f"\n输入形状:")
    print(f"  h_new: {h_new.shape}")
    print(f"  delta_pos_order_l: {delta_pos_order_l.shape}")
    
    # 当前实现的输出
    with torch.no_grad():
        current_output = current_tp(h_new, delta_pos_order_l)
        current_output_sum = torch.sum(current_output, dim=1)
    
    print(f"\n当前实现输出形状:")
    print(f"  tensor_product 输出: {current_output.shape}")
    print(f"  求和后输出: {current_output_sum.shape}")
    print(f"  输出统计:")
    print(f"    mean: {current_output_sum.mean().item():.6f}")
    print(f"    std: {current_output_sum.std().item():.6f}")
    print(f"    min: {current_output_sum.min().item():.6f}")
    print(f"    max: {current_output_sum.max().item():.6f}")
    
    # ========== e3nn 实现 ==========
    print("\n" + "=" * 80)
    print("2. e3nn TensorProduct 实现")
    print("=" * 80)
    
    e3nn_tp = create_e3nn_tensor_product(
        irreps_in, irreps_out, order, 
        learnable_weight=True, 
        connection_mode="uvw"
    )
    
    # 复制权重（如果可能）
    if hasattr(current_tp, 'weight') and hasattr(e3nn_tp, 'weight'):
        if current_tp.weight.shape == e3nn_tp.weight.shape:
            e3nn_tp.weight.data.copy_(current_tp.weight.data)
            print("✓ 成功复制权重")
        else:
            print(f"⚠ 权重形状不匹配:")
            print(f"  当前实现: {current_tp.weight.shape}")
            print(f"  e3nn: {e3nn_tp.weight.shape}")
    
    # e3nn TensorProduct 的输出
    # 需要将输入 reshape 为 e3nn 期望的扁平化格式
    # 当前实现: (f_N1, f_N2, (order+1)**2, channels) -> e3nn: (f_N1, f_N2, irreps_in1.dim)
    irreps_in_obj = o3.Irreps(irreps_in)
    irreps_edge_obj = o3.Irreps(f"1x{order}e")
    
    # h_new: (f_N1, f_N2, (order+1)**2, head*hidden) -> (f_N1, f_N2, irreps_in1.dim)
    h_new_flat = h_new.reshape(f_N1, f_N2, -1)
    # delta_pos_order_l: (f_N1, f_N2, (order+1)**2, 1) -> (f_N1, f_N2, irreps_edge.dim)
    delta_pos_order_l_flat = delta_pos_order_l.squeeze(-1)  # (f_N1, f_N2, (order+1)**2)
    
    print(f"\n输入 reshape 信息:")
    print(f"  h_new 原始形状: {h_new.shape}")
    print(f"  h_new 扁平化后: {h_new_flat.shape}")
    print(f"  irreps_in1.dim: {irreps_in_obj.dim}")
    print(f"  delta_pos_order_l 原始形状: {delta_pos_order_l.shape}")
    print(f"  delta_pos_order_l 扁平化后: {delta_pos_order_l_flat.shape}")
    print(f"  irreps_edge.dim: {irreps_edge_obj.dim}")
    
    with torch.no_grad():
        e3nn_output = e3nn_tp(h_new_flat, delta_pos_order_l_flat)
        e3nn_output_sum = torch.sum(e3nn_output, dim=1)
    
    print(f"\ne3nn 实现输出形状:")
    print(f"  tensor_product 输出: {e3nn_output.shape}")
    print(f"  求和后输出: {e3nn_output_sum.shape}")
    print(f"  输出统计:")
    print(f"    mean: {e3nn_output_sum.mean().item():.6f}")
    print(f"    std: {e3nn_output_sum.std().item():.6f}")
    print(f"    min: {e3nn_output_sum.min().item():.6f}")
    print(f"    max: {e3nn_output_sum.max().item():.6f}")




if __name__ == "__main__":
    # 运行基本测试
    test_tensor_product_replacement()
    
    print("\n" + "=" * 80)
    print("所有测试完成")
    print("=" * 80)
    print("\n提示: 查看 test_e3nn_replacement_README.md 了解如何解读测试结果")

