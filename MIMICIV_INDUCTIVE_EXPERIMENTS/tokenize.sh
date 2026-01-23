#!/usr/bin/env bash

# This makes the script fail if any internal script fails
set -e

# Function to display help message
function display_help() {
    echo "Usage: $0 <MEDS_DIR> <MODEL_DIR> <N_WORKERS> <PIPELINE_CONFIG_PATH> [additional args]"
    echo
    echo "This script processes MIMIC-IV data through the MEDS transformation pipeline."
    echo
    echo "Arguments:"
    echo "  MEDS_DIR                              Input directory for MEDS data."
    echo "  MODEL_DIR                             Output directory for processed data."
    echo "  N_WORKERS                             Number of parallel workers for processing."
    echo "  PIPELINE_CONFIG_PATH                  Pipeline configuration file."
    echo "  [additional args]                     Additional arguments passed to MEDS_transform-pipeline."
    echo
    echo "Options:"
    echo "  -h, --help          Display this help message and exit."
    exit 1
}

# Check if the first parameter is '-h' or '--help'
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    display_help
fi

# Check for mandatory parameters
if [ "$#" -lt 4 ]; then
    echo "Error: Incorrect number of arguments provided."
    display_help
fi

export MEDS_DIR="$1"
export MODEL_DIR="$2"
export N_WORKERS="$3"
export PIPELINE_CONFIG_PATH="$4"

echo "MEDS_DIR: $MEDS_DIR"
echo "MODEL_DIR: $MODEL_DIR"
echo "N_WORKERS: $N_WORKERS"
echo "PIPELINE_CONFIG_PATH: $PIPELINE_CONFIG_PATH"

shift 4

echo "Running extraction pipeline."
MEDS_transform-pipeline "$PIPELINE_CONFIG_PATH" "$@"

# Run custom tokenization step
python -m meds_torch.utils.custom_tokenization \
    dataset.root_dir="$MEDS_DIR" \
    output_dir="$MODEL_DIR/tokenization" \
    stages=[tokenization] \
    stage=tokenization \
    stage_cfg.data_input_dir="$MODEL_DIR/data" \
    stage_cfg.output_dir="$MODEL_DIR/tokenization"

