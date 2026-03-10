export PYTHONPATH="/pcpt/pcpt/project/jiangjunyuan/code/VLA/Drive_pi:$PYTHONPATH"
export LOG_DIR="/pcpt/pcpt/project/jiangjunyuan/code/VLA/Drive_pi/logs"
BASE_CONFIG="qwen3.yaml"
python script/run.py \
  --config-path=../config/train/DrivePi0 \
  --config-name=${BASE_CONFIG} 