#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
代码执行路径追踪工具

这个脚本展示了如何追踪 `output = model(data)` 的执行路径。
可以通过以下方式使用：
1. 使用 Python 调试器 (pdb)
2. 使用 trace 模块
3. 使用装饰器添加日志
4. 使用 IDE 的调试功能
cd /home/v-hanhaozhi/1-eq-learn/E2Former && python trace_model_execution.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# 方法 1: 使用 Python 调试器 (pdb)
# ============================================================================
def trace_with_pdb():
    """
    使用 pdb 调试器追踪代码执行
    
    使用方法：
    1. 在代码中添加：import pdb; pdb.set_trace()
    2. 或者在命令行运行：python -m pdb infer-hhz.py
    """
    print("=" * 60)
    print("方法 1: 使用 pdb 调试器")
    print("=" * 60)
    print("""
    在 infer-hhz.py 的第 186 行之前添加：
    
    import pdb; pdb.set_trace()
    output = model(data)
    
    然后运行脚本，程序会在该行暂停，你可以：
    - n (next): 执行下一行
    - s (step): 进入函数内部
    - c (continue): 继续执行
    - l (list): 显示当前代码
    - w (where): 显示调用栈
    """)


# ============================================================================
# 方法 2: 使用 trace 模块
# ============================================================================
def trace_with_trace_module():
    """
    使用 Python trace 模块追踪所有函数调用
    
    使用方法：
    python -m trace --trace infer-hhz.py
    """
    print("=" * 60)
    print("方法 2: 使用 trace 模块")
    print("=" * 60)
    print("""
    在命令行运行：
    python -m trace --trace infer-hhz.py
    
    这会显示所有执行的代码行。
    
    或者只追踪函数调用：
    python -m trace --trackcalls infer-hhz.py
    """)


# ============================================================================
# 方法 3: 使用装饰器添加日志
# ============================================================================
import functools
import traceback

def trace_calls(func):
    """装饰器：追踪函数调用"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"\n{'='*60}")
        print(f"调用函数: {func.__name__}")
        print(f"文件: {func.__code__.co_filename}")
        print(f"行号: {func.__code__.co_firstlineno}")
        print(f"{'='*60}")
        
        # 打印调用栈
        stack = traceback.extract_stack()
        print("\n调用栈:")
        for frame in stack[-5:-1]:  # 显示最近 4 层调用栈
            print(f"  {frame.filename}:{frame.lineno} in {frame.name}")
        
        # 调用原函数
        result = func(*args, **kwargs)
        print(f"函数 {func.__name__} 执行完成\n")
        return result
    return wrapper


# ============================================================================
# 方法 4: 代码执行路径文档
# ============================================================================
def print_execution_path():
    """
    打印代码执行路径的文档说明
    """
    print("=" * 60)
    print("代码执行路径文档")
    print("=" * 60)
    
    execution_path = """
    output = model(data) 的执行路径：
    
    1. infer-hhz.py:186
       └─> model(data)
           └─> E2FormerBackbone.forward() [E2Former_wrapper.py:459]
               └─> self.forward_fn(data, ...) [E2Former_wrapper.py:472]
                   └─> E2FormerBackbone.compiled_forward() [E2Former_wrapper.py:282]
                       │
                       ├─> process_batch_data() [E2Former_wrapper.py:28]
                       │   └─> 处理批次数据，转换为模型期望的格式
                       │
                       ├─> 处理周期性边界条件 (PBC)
                       │   └─> cell_expander.expand_includeself() [如果启用 PBC]
                       │
                       ├─> 生成 token embedding
                       │   └─> self.embedding(atomic_numbers) [E2Former_wrapper.py:341]
                       │
                       ├─> 调用 E2Former 解码器
                       │   └─> self.decoder(...) [E2Former_wrapper.py:356]
                       │       └─> E2former.forward() [e2former_main.py:382]
                       │           │
                       │           ├─> 数据准备和预处理 [e2former_main.py:434-457]
                       │           │   └─> 提取位置、掩码等
                       │           │
                       │           ├─> 处理周期性边界条件 [e2former_main.py:523-555]
                       │           │   └─> 扩展周期性图像
                       │           │
                       │           ├─> 构建邻居图 [e2former_main.py:563-580]
                       │           │   └─> construct_radius_neighbor()
                       │           │
                       │           ├─> 原子嵌入 [e2former_main.py:586-595]
                       │           │   └─> unifiedtokentoembedding() 或 default_node_embedding()
                       │           │
                       │           ├─> 计算球谐函数 [e2former_main.py:600-614]
                       │           │   └─> get_powers()
                       │           │
                       │           ├─> 边度嵌入 [e2former_main.py:620-640]
                       │           │   └─> edge_deg_embed_dense()
                       │           │
                       │           ├─> Transformer 块前向传播 [e2former_main.py:653-667]
                       │           │   └─> 循环遍历 self.blocks
                       │           │       └─> TransBlock.forward()
                       │           │           ├─> E2AttentionArbOrder_sparse (注意力层)
                       │           │           ├─> MessageBlock (消息传递)
                       │           │           └─> FeedForwardNetwork (前馈网络)
                       │           │
                       │           └─> 最终归一化和特征提取 [e2former_main.py:679-736]
                       │               └─> 提取标量特征 (能量) 和向量特征 (力)
                       │
                       └─> 展平节点特征 [E2Former_wrapper.py:377]
                           └─> flatten_node_features() [E2Former_wrapper.py:404]
    
    返回结果：
    {
        "node_features": 标量特征 (能量相关),
        "node_vec_features": 向量特征 (力相关),
        "node_irreps": 不可约表示,
        "node_irreps_his": 历史不可约表示,
        ...
    }
    """
    print(execution_path)


# ============================================================================
# 方法 5: 使用 sys.settrace 追踪
# ============================================================================
def trace_calls_and_lines(frame, event, arg):
    """追踪函数调用和代码行执行"""
    if event == 'call':
        filename = frame.f_code.co_filename
        func_name = frame.f_code.co_name
        line_no = frame.f_lineno
        
        # 只追踪项目内的文件
        if 'E2Former' in filename or 'infer-hhz' in filename:
            print(f"调用: {func_name}() 在 {filename}:{line_no}")
    
    elif event == 'line':
        filename = frame.f_code.co_filename
        line_no = frame.f_lineno
        
        # 只追踪关键文件
        if 'E2Former' in filename or 'infer-hhz' in filename:
            print(f"执行: {filename}:{line_no}")
    
    return trace_calls_and_lines


def enable_trace():
    """启用追踪"""
    sys.settrace(trace_calls_and_lines)


def disable_trace():
    """禁用追踪"""
    sys.settrace(None)


# ============================================================================
# 主函数
# ============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("代码执行路径追踪工具")
    print("=" * 60 + "\n")
    
    # 打印执行路径文档
    print_execution_path()
    
    print("\n" + "=" * 60)
    print("使用方法")
    print("=" * 60 + "\n")
    
    # 方法 1
    trace_with_pdb()
    
    # 方法 2
    trace_with_trace_module()
    
    print("\n" + "=" * 60)
    print("方法 3: 使用装饰器")
    print("=" * 60)
    print("""
    在关键函数上添加 @trace_calls 装饰器，例如：
    
    from trace_model_execution import trace_calls
    
    @trace_calls
    def forward(self, data):
        ...
    """)
    
    print("\n" + "=" * 60)
    print("方法 4: 使用 sys.settrace")
    print("=" * 60)
    print("""
    在 infer-hhz.py 中添加：
    
    from trace_model_execution import enable_trace, disable_trace
    
    enable_trace()
    output = model(data)
    disable_trace()
    """)
    
    print("\n" + "=" * 60)
    print("方法 5: 使用 IDE 调试器")
    print("=" * 60)
    print("""
    在 VS Code / PyCharm 中：
    1. 在第 186 行设置断点
    2. 使用 F5 启动调试
    3. 使用 F10 (step over) 或 F11 (step into) 单步执行
    4. 查看调用栈窗口了解函数调用关系
    """)
