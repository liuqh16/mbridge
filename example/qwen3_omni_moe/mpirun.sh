#!/bin/bash
# Multi-node test script using MPI

HOSTFILE=$1

mpirun --allow-run-as-root \
    -np 2 \
    --hostfile ${HOSTFILE} \
    -bind-to none \
    -map-by slot \
    -mca pml ob1 \
    -mca btl ^openib \
    -x NCCL_DEBUG=INFO \
    -x NCCL_SOCKET_IFNAME=bond1 \
    -x GLOO_SOCKET_IFNAME=bond1 \
    bash example/qwen3_omni_moe/run_test.sh

