#!/bin/bash
# Health monitor for the random-ordering runs.
# Prints: active job states, per-experiment training progress, error flags,
# and finished-job outcomes (via sacct) with checkpoint readiness for eval.
LOGDIR=/fsx/zyhang/verl/logs/ebpo
CKPT=/checkpoints/zyhang/ebpo
USER_NAME=zyhang

echo "===================== $(date) ====================="
echo "--- ACTIVE JOBS ---"
squeue -u "$USER_NAME" -o "%.10i %.48j %.9T %.6M %R" | grep -E "random|JOBID"

echo ""
echo "--- PROGRESS / ERRORS (per experiment log) ---"
for f in "$LOGDIR"/*random*.log; do
    [ -e "$f" ] || continue
    exp=$(basename "$f" .log)
    step=$(grep -aoE "[0-9]+/204" "$f" 2>/dev/null | tail -1)
    err=$(grep -aiE "out of memory|CUDA out of memory|OutOfMemory|Traceback|RuntimeError|NCCL.*error|AssertionError|CUDA error" "$f" 2>/dev/null | tail -1)
    printf "%-58s %s\n" "$exp" "step ${step:-0/204}"
    [ -n "$err" ] && printf "    !! ERROR: %s\n" "$err"
done

echo ""
echo "--- FINISHED JOBS (of the 14 real runs) ---"
JOBS="1562590,1562591,1562659,1562660,1562661,1562662,1562663,1562664,1562665,1562666,1562667,1562668,1562669,1562670"
sacct -j "$JOBS" --format=JobID,JobName%48,State,Elapsed,ExitCode -X 2>/dev/null \
    | grep -viE "RUNNING|PENDING|\.batch"

echo ""
echo "--- CHECKPOINTS AVAILABLE ---"
for d in "$CKPT"/*random*/; do
    [ -d "$d" ] || continue
    exp=$(basename "$d")
    latest=$(find "$d" -maxdepth 1 -type d -name "global_step_*" 2>/dev/null | sort -t_ -k3 -n | tail -1)
    if [ -n "$latest" ]; then
        merged="no"
        compgen -G "$latest/actor/huggingface/*.safetensors" >/dev/null 2>&1 && merged="yes(HF)"
        printf "%-58s %s  merged=%s\n" "$exp" "$(basename "$latest")" "$merged"
    fi
done
echo "==================================================="
