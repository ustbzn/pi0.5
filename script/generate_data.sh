#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_DIR"

# ===============================================
# Data Generation Script for B2D Dataset
# ===============================================

# Directory Configuration
WORK_DIR="${WORK_DIR:-/pcpt/pcpt/project/jiangjunyuan/code/VLA/DriveMoE}"     # Directory for processed data cache
DATASET_PATH="${DATASET_PATH:-/D2/NXT/vlm/dataset/Bench2Drive-Base}"          # Directory containing original B2D dataset
CAMERA_LABEL_DIR="${CAMERA_LABEL_DIR:-data/camera_labels}"                    # Directory containing camera labels
SCENARIO_LABEL_DIR="${SCENARIO_LABEL_DIR:-data/scenario_labels}"              # Directory containing scenario labels

# Temporal Configuration
WINDOW_SIZE=5                 # Number of historical frames/observations used as input context
HORIZON_SIZE=20               # Number of future trajectory points to predict

# Important Note on HORIZON_SIZE:
# --------------------------------------
# HORIZON_SIZE determines how many future timesteps the model will predict.
# 
# - For OPEN-LOOP evaluation (testing with ground truth history):
#   MUST be set to 20 to ensure fair comparison with baseline methods.
#   This is because standard benchmarks evaluate exactly 20 future steps.
#
# - For CLOSED-LOOP evaluation (testing with predicted history):
#   Can be set to any value based on your requirements.
#   There is no restriction in closed-loop settings.
# --------------------------------------

echo "==============================================="
echo "B2D Data Generation Pipeline"
echo "==============================================="
echo "Processed data cache dir: $WORK_DIR"
echo "Origin b2d data dir: $DATASET_PATH"
echo "History size (input context): $WINDOW_SIZE"
echo "Horizon size (future predictions): $HORIZON_SIZE"
echo ""
echo "NOTE: Horizon size specifies the number of future trajectory points"
echo "      to predict. For OPEN-LOOP testing, MUST use 20 for fair comparison."
echo "      For CLOSED-LOOP testing, no restrictions apply."
echo "==============================================="

for required_dir in "$DATASET_PATH" "$CAMERA_LABEL_DIR" "$SCENARIO_LABEL_DIR"; do
    if [ ! -d "$required_dir" ]; then
        echo "Error: Required directory not found: $required_dir"
        echo "Please download Bench2Drive data and DriveMoE labels, or override paths with:"
        echo "  DATASET_PATH=/path/to/Bench2Drive-Base \\"
        echo "  CAMERA_LABEL_DIR=/path/to/camera_labels \\"
        echo "  SCENARIO_LABEL_DIR=/path/to/scenario_labels \\"
        echo "  bash script/generate_data.sh"
        exit 1
    fi
done

# Create working directory
mkdir -p "$WORK_DIR" || { 
    echo "Error: Failed to create directory $WORK_DIR"
    exit 1
}

# Generate action data
echo "Step 1: Generating action data..."
python src/data/generate_data/generate_action.py \
    --b2d_dataset_dir "$DATASET_PATH" \
    --camera_label_dir "$CAMERA_LABEL_DIR" \
    --scenario_label_dir "$SCENARIO_LABEL_DIR" \
    --work_dir "$WORK_DIR"

# Create windowed sequences
echo "Step 2: Creating windowed sequences..."
python src/data/generate_data/window.py \
    --work_dir "$WORK_DIR" \
    --window_size "$WINDOW_SIZE" \
    --horizon "$HORIZON_SIZE" \
    --num_cpus 32

# Compute dataset statistics
echo "Step 3: Computing dataset statistics..."
python src/data/generate_data/get_statistics.py \
    --data_path "${WORK_DIR}/b2d_action/train"

# File paths for statistics JSON
TARGET_JSON="${PWD}/config/statistics/b2d_statistics.json"
STATISTICS_DIR="${PWD}/config/statistics"
GENERATED_JSON="${PWD}/b2d_statistics.json"

# Move generated statistics file
if [ -f "$GENERATED_JSON" ]; then
    if [ ! -f "$TARGET_JSON" ]; then
        echo "Step 4: Moving statistics file to target directory..."
        mv "$GENERATED_JSON" "$STATISTICS_DIR" || {
            echo "Error: Failed to move file"
            exit 1
        }
        echo "Data generation completed successfully!"
    else
        echo "Warning: Target file already exists at $TARGET_JSON (skipped)"
        echo "$(date) - WARNING: File $TARGET_JSON already exists" >> "$WORK_DIR/process.log"
    fi
else
    echo "Error: Generated JSON file not found at $GENERATED_JSON"
    exit 1
fi

echo "==============================================="
echo "Pipeline execution completed"
echo "==============================================="
