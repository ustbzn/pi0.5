#!/bin/sh
export TZ='UTC-8'

# 第一阶段：并行执行文件夹 1-4
echo "Launching parallel jobs for folders 1-4 ..."

for i in $(seq 1 4); do
    (
        echo "=========================================="
        echo "Processing folder $i ..."
        echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "=========================================="
        nuscene-dataset-tools/run_rest_parallel.sh $i
        echo ""
        echo "Folder $i completed at $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""
    ) &
done

wait
echo "Folders 1-4 processing completed!"

# 第二阶段：并行执行文件夹 5-8
echo "Launching parallel jobs for folders 5-8 ..."

for i in $(seq 5 8); do
    (
        echo "=========================================="
        echo "Processing folder $i ..."
        echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "=========================================="
        nuscene-dataset-tools/run_rest_parallel.sh $i
        echo ""
        echo "Folder $i completed at $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""
    ) &
done

wait
echo "Folders 5-8 processing completed!"

# 第三阶段：并行执行文件夹 9-10
echo "Launching parallel jobs for folders 9-10 ..."

for i in $(seq 9 10); do
    (
        echo "=========================================="
        echo "Processing folder $i ..."
        echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "=========================================="
        nuscene-dataset-tools/run_rest_parallel.sh $i
        echo ""
        echo "Folder $i completed at $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""
    ) &
done

wait
echo "All folders (1-10) processing completed with 4,4,2 grouping (parallel)!"