#!/bin/bash
ps -ef | grep python | awk  '{print $2}' | xargs -I {} kill -9 {}
sleep 1

DIR="$(cd "$( dirname "$0" )" && pwd)"
# mbridge
cd ${DIR}/../..

export PYTHONPATH=$DIR/../..:$DIR/../../../Megatron-LM:$PYTHONPATH
echo "PYTHONPATH ${PYTHONPATH}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HF_DATASETS_OFFLINE=1
export GLOO_SOCKET_IFNAME=bond1
export NCCL_SOCKET_IFNAME=bond1

readonly GPUS_PER_NODE=8
readonly NODE_RANK="${OMPI_COMM_WORLD_RANK:-0}"
readonly NNODES="${OMPI_COMM_WORLD_SIZE:-1}"
readonly WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
readonly MASTER_PORT=65535
export MASTER_ADDR="${_MASTER_ADDR:-localhost}"

readonly TP_SIZE=2
readonly PP_SIZE=2
readonly CP_SIZE=2
readonly EP_SIZE=2

echo "INFO
__POD_IP__ $__POD_IP__
NODE_RANK $NODE_RANK
NNODES $NNODES
TP_SIZE $TP_SIZE
PP_SIZE $PP_SIZE
CP_SIZE $CP_SIZE
EP_SIZE $EP_SIZE
"

# torch 启动参数
DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"

# ========== Test 1: Image Only (like Qwen3-VL-MoE) ==========
echo "=========================================="
echo "Test 1: Qwen3OmniMoe with IMAGE ONLY"
echo "=========================================="

# Run HF forward pass (image only)
python example/qwen3_omni_moe/hf_fwd_omni.py \
    --model_path ../hf-hub/Qwen/Qwen3-Omni-MoE-Instruct \
    --use_image

# Run mcore forward pass (image only)
torchrun $DISTRIBUTED_ARGS \
    example/qwen3_omni_moe/load_model_and_forward.py \
    --tp $TP_SIZE \
    --pp $PP_SIZE \
    --ep $EP_SIZE \
    --etp 1 \
    --cp $CP_SIZE \
    --model_path ../hf-hub/Qwen/Qwen3-Omni-MoE-Instruct \
    --use_image \
    --check_with_hf \
    --check_export

# ========== Test 2: Audio + Image (Full Multimodal) ==========
echo "=========================================="
echo "Test 2: Qwen3OmniMoe with AUDIO + IMAGE"
echo "=========================================="

# Run HF forward pass (audio + image)
python example/qwen3_omni_moe/hf_fwd_omni.py \
    --model_path ../hf-hub/Qwen/Qwen3-Omni-MoE-Instruct \
    --use_audio \
    --use_image

# Run mcore forward pass (audio + image)
torchrun $DISTRIBUTED_ARGS \
    example/qwen3_omni_moe/load_model_and_forward.py \
    --tp $TP_SIZE \
    --pp $PP_SIZE \
    --ep $EP_SIZE \
    --etp 1 \
    --cp $CP_SIZE \
    --model_path ../hf-hub/Qwen/Qwen3-Omni-MoE-Instruct \
    --use_audio \
    --use_image \
    --check_with_hf

# ========== Test 3: Audio Only ==========
echo "=========================================="
echo "Test 3: Qwen3OmniMoe with AUDIO ONLY"
echo "=========================================="

# Run HF forward pass (audio only)
python example/qwen3_omni_moe/hf_fwd_omni.py \
    --model_path ../hf-hub/Qwen/Qwen3-Omni-MoE-Instruct \
    --use_audio

# Run mcore forward pass (audio only)
torchrun $DISTRIBUTED_ARGS \
    example/qwen3_omni_moe/load_model_and_forward.py \
    --tp $TP_SIZE \
    --pp $PP_SIZE \
    --ep $EP_SIZE \
    --etp 1 \
    --cp $CP_SIZE \
    --model_path ../hf-hub/Qwen/Qwen3-Omni-MoE-Instruct \
    --use_audio \
    --check_with_hf

echo "=========================================="
echo "✓ All Qwen3OmniMoe tests completed!"
echo "=========================================="

