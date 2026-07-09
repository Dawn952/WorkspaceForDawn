export HCCL_OP_EXPANSION_MODE="AIV"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_MLAPO=1

export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

export HCCL_BUFFSIZE=800
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_BALANCE_SCHEDULING=1


TARGET_MODEL="/mnt/weight/Kimi-K2_5-w4a8c8"
DRAFT_MODEL="/mnt/weight/kimi_k2.5_dflash"
EAGLE_MODEL="/mnt/weight/k2.6_eagle"

vllm serve /mnt/weight/Kimi-K2_5-w4a8c8 \
    --quantization ascend \
    --enable-auto-tool-choice \
    --tool-call-parser kimi_k2 \
    --reasoning-parser kimi_k2 \
    --allowed-local-media-path / \
    --served-model-name kimi_k26 \
    --trust-remote-code \
    --tensor-parallel-size 16 \
    --data-parallel-size 1 \
    --no-enable-prefix-caching \
    --enable-expert-parallel \
    --port 8089 \
    --max-num-seqs 8 \
    --max-model-len 131072 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.88 \
    --seed 42 \
    --async-scheduling \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --profiler-config '{"profiler": "torch", "torch_profiler_dir": "./vllm_profile", "torch_profiler_with_stack": true}' \
    --mm-processor-cache-gb 0 \
    --mm-encoder-tp-mode data \
    --speculative-config '{"method": "dflash","model": "/mnt/weight/kimi_k2.5_dflash", "num_speculative_tokens": 15}'
