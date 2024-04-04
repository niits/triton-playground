# !/bin/bash

set -eauxo pipefail;

# Make tensorrt folder
if [ ! -d "tensorrt" ]; then
    mkdir tensorrt
fi;

# Make log folder that contains current time

LOG_DIR="logs/$(date +'%Y-%m-%d_%H-%M-%S')"
mkdir -p $LOG_DIR

echo "Converting VAE model to TensorRT engine if not already done..."

if [ ! -f "tensorrt/vae.plan" ]; then
    /usr/src/tensorrt/bin/trtexec \
    --onnx=onnx_models/vae.onnx \
    --saveEngine=tensorrt/vae.plan \
    --explicitBatch  \
    --verbose \
    > $LOG_DIR/vae_conversion.log
    ;
else
    echo "VAE model already converted to TensorRT engine."
fi

echo "Converting text encoder model to TensorRT engine if not already done.."

if [ ! -f "tensorrt/text_encoder.plan" ]; then
    /usr/src/tensorrt/bin/trtexec \
    --onnx=onnx_models/text_encoder.onnx \
    --saveEngine=tensorrt/text_encoder.plan \
    --explicitBatch \
    --verbose \
    > $LOG_DIR/text_encoder_conversion.log
else
    echo "Text encoder model already converted to TensorRT engine."
fi

MODEL_REGISTRY_DIR="/mnt/models"

function move_to_model_registry {
    if [ ! -d "$MODEL_REGISTRY_DIR/$1" ]; then
        mkdir $MODEL_REGISTRY_DIR/$1
    fi;

    # Get last version from name of all children folders
    LAST_VERSION=$(ls $MODEL_REGISTRY_DIR/$1 | sort -n | tail -n 1)

    # Increment version number
    NEW_VERSION=$((LAST_VERSION + 1))

    # Make folder for new version
    mkdir $MODEL_REGISTRY_DIR/$1/$NEW_VERSION

    # Move plan file into new version folder
    mv tensorrt/$1.plan $MODEL_REGISTRY_DIR/$1/$NEW_VERSION/$1.plan
}

echo "Moving converted models to model registry..."

move_to_model_registry "vae"

move_to_model_registry "text_encoder"

echo "Done."