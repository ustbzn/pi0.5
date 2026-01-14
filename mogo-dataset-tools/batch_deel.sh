#!/bin/bash

# 批量处理 bag 文件的脚本
# 用法: ./batch_deel.sh <bag_root_directory> [script_path]
# 示例: ./batch_deel.sh /path/to/bag/root

set -e  # 遇到错误立即退出

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/parseRosBag-2O.py"

# 检查参数
if [ $# -lt 1 ]; then
    echo "用法: $0 <bag_root_directory> [python_script_path]"
    echo "示例: $0 /path/to/bag/root"
    exit 1
fi

BAG_ROOT="$1"

# 如果提供了第二个参数，使用它作为 Python 脚本路径
if [ $# -ge 2 ]; then
    PYTHON_SCRIPT="$2"
fi

# 检查根目录是否存在
if [ ! -d "$BAG_ROOT" ]; then
    echo "[ERROR] 目录不存在: $BAG_ROOT"
    exit 1
fi

# 检查 Python 脚本是否存在
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "[ERROR] Python 脚本不存在: $PYTHON_SCRIPT"
    exit 1
fi

# 统计变量
TOTAL=0
SUCCESS=0
FAILED=0
FAILED_DIRS=()

echo "=========================================="
echo "开始批量处理 bag 文件"
echo "根目录: $BAG_ROOT"
echo "Python 脚本: $PYTHON_SCRIPT"
echo "=========================================="
echo ""

# 查找所有包含 .bag 文件的目录
while IFS= read -r -d '' bag_dir; do
    # 检查该目录是否包含 master 和 slave bag 文件
    has_master=false
    has_slave=false
    
    for bag_file in "$bag_dir"/*.bag; do
        if [ -f "$bag_file" ]; then
            filename=$(basename "$bag_file")
            if [[ "$filename" == *"-rosmaster-"* ]]; then
                has_master=true
            elif [[ "$filename" == *"-rosslave-"* ]]; then
                has_slave=true
            fi
        fi
    done
    
    # 如果同时有 master 和 slave，才进行处理
    if [ "$has_master" = true ] && [ "$has_slave" = true ]; then
        TOTAL=$((TOTAL + 1))
        echo "[$TOTAL] 处理目录: $bag_dir"
        
        # 切换到 bag 目录的父目录作为工作目录（这样输出文件会在 bag 目录同级）
        bag_parent=$(dirname "$bag_dir")
        bag_name=$(basename "$bag_dir")
        
        # 执行 Python 脚本
        if python "$PYTHON_SCRIPT" "$bag_dir" 2>&1 | tee "${bag_dir}/process.log"; then
            SUCCESS=$((SUCCESS + 1))
            echo "[✓] 成功处理: $bag_dir"
        else
            FAILED=$((FAILED + 1))
            FAILED_DIRS+=("$bag_dir")
            echo "[✗] 处理失败: $bag_dir"
        fi
        echo ""
    else
        echo "[SKIP] 跳过目录（缺少 master 或 slave bag）: $bag_dir"
    fi
done < <(find "$BAG_ROOT" -type d -print0)

# 输出统计信息
echo "=========================================="
echo "批量处理完成"
echo "总计: $TOTAL 个目录"
echo "成功: $SUCCESS 个"
echo "失败: $FAILED 个"
echo "=========================================="

# 如果有失败的目录，列出它们
if [ $FAILED -gt 0 ]; then
    echo ""
    echo "失败的目录列表:"
    for dir in "${FAILED_DIRS[@]}"; do
        echo "  - $dir"
    done
    exit 1
fi

exit 0

