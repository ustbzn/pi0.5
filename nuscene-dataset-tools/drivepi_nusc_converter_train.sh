# export CUDA_VISIBLE_DEVICES=0,1
#export PYTHONPATH=/path/to/your/project/root:$PYTHONPATH
export TZ='UTC-8'   # 北京时间

folder=$1
root_path=/D2/NXT/vlm/dataset/nuScenes/nuscene-all/$folder
out_path=datasets/nuscene-train-pi0-all-0109-1/$folder
mkdir -p $out_path

echo "Start generating dataset for folder $folder..."
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"   # 开始时间

python nuscene-dataset-tools/drivepi_nusc_data_converter.py \
    nuscenes \
    --root-path  $root_path \
    --out-dir  $out_path \
    --extra-tag drivepi_nusc \
    --canbus /D2/NXT/vlm/dataset/nuScenes/can_bus_nuscene/ \
    --version v1.0

echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"     # 结束时间
echo "Dataset generation $folder finished."