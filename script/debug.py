# debug_import.py
import sys
import os

print("=== 完整的导入诊断 ===")
print(f"当前工作目录: {os.getcwd()}")
print(f"脚本所在目录: {os.path.dirname(os.path.abspath(__file__))}")

print("\n=== sys.path 内容 ===")
for i, path in enumerate(sys.path):
    mark = "← 当前目录" if path == os.getcwd() else ""
    mark = "← 脚本目录" if path == os.path.dirname(os.path.abspath(__file__)) else mark
    print(f"{i:2}: {path} {mark}")

print("\n=== 尝试导入 train 模块 ===")
try:
    # 检查可能的导入路径
    possible_paths = [
        "/pcpt/pcpt/project/jiangjunyuan/code/VLA/Drive_pi/src/agent/drivepi0/train.py",
        "/D2/NXT/vlm/miaobinqi/DriveMoE/src/agent/drivepi0/train.py",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ 文件存在: {path}")
        else:
            print(f"❌ 文件不存在: {path}")
    
    # 实际导入
    import importlib.util
    
    for path in possible_paths:
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location("train_module", path)
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
                print(f"✅ 可以直接导入: {path}")
                break
            except Exception as e:
                print(f"❌ 导入失败 {path}: {e}")
    
    # 正常导入
    print("\n=== 正常导入测试 ===")
    from src.agent.drivepi0 import train
    print(f"✅ 正常导入成功: {train.__file__}")
    
except Exception as e:
    print(f"❌ 导入失败: {e}")
    import traceback
    traceback.print_exc()

print("\n=== 检查已安装的包 ===")
# 检查是否有包冲突
import pkgutil
for importer, modname, ispkg in pkgutil.iter_modules():
    if "drivepi" in modname.lower() or "drive" in modname.lower():
        print(f"找到相关模块: {modname}")

print("\n=== 检查模块缓存 ===")
import importlib
if 'src.agent.drivepi0.train' in sys.modules:
    print("train 模块已在缓存中")
    print(f"缓存位置: {sys.modules['src.agent.drivepi0.train'].__file__}")