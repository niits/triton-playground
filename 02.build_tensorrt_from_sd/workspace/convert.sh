# !/bin/bash

#/bin/bash

set -eou pipefail;

mkdir -p tensorrt_engines;

trtexec --onnx=/mnt/models/onnx/stable-diffusion-v1-5/text_encoder/model.onnx \
    --minShapes=input_ids:1x77 \
    --optShapes=input_ids:4x77 \
    --maxShapes=input_ids:16x77 \
    --shapes=input_ids:2x77 \
    --saveEngine=tensorrt_engines/text_encoder.engine \
    --workspace=600;


trtexec --onnx=/mnt/models/onnx/stable-diffusion-v1-5/vae_decoder/model.onnx \
    --shapes=latent_sample:2x4x64x64 \
    --saveEngine=tensorrt_engines/vae_decoder.engine \
    --workspace=1400 \
    --verbose;

trtexec --onnx=/mnt/models/onnx/unet.onnx \
    --shapes=sample:2x4x64x64,timestep:2x1,encoder_hidden_states:2x77x768 \
    --saveEngine=tensorrt_engines/unet.engine \
    --verbose  \
    --workspace=8000;
