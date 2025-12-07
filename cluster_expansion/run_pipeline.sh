#!/bin/bash
#
# Run the full Cluster Expansion pipeline
#
# Usage:
#     bash run_pipeline.sh [OPTIONS]
#
# Options:
#     --skip-step0    Skip local environment extraction
#     --skip-step1    Skip MACE energy computation
#     --skip-step2    Skip wrangler processing
#     --skip-step3    Skip ECI fitting
#

set -e  # Exit on error

# Default parameters
N_WORKERS=32
BATCH_SIZE=32
GPU_IDX=4
MACE_MODEL="medium-omat-0"
N_RANDOM=5000
TEST_RATIO=0.2

# Parse arguments
SKIP_STEP0=false
SKIP_STEP1=false
SKIP_STEP2=false
SKIP_STEP3=false

for arg in "$@"; do
    case $arg in
        --skip-step0) SKIP_STEP0=true ;;
        --skip-step1) SKIP_STEP1=true ;;
        --skip-step2) SKIP_STEP2=true ;;
        --skip-step3) SKIP_STEP3=true ;;
    esac
done

echo "============================================================"
echo "Cluster Expansion Pipeline"
echo "============================================================"
echo "Workers: $N_WORKERS"
echo "GPU: $GPU_IDX"
echo "MACE Model: $MACE_MODEL"
echo "============================================================"
echo ""

# Step 0: Extract local environments
if [ "$SKIP_STEP0" = false ]; then
    echo "[Step 0] Extracting local environments..."
    python 0_extract_local_envs_parallel.py \
        --n_workers $N_WORKERS \
        --max_vac_per_file 50 \
        --n_random $N_RANDOM \
        --test_ratio $TEST_RATIO
    echo "[Step 0] Done!"
    echo ""
else
    echo "[Step 0] Skipped"
fi

# Step 1: Compute MACE energies
if [ "$SKIP_STEP1" = false ]; then
    echo "[Step 1] Computing MACE energies..."
    python 1_compute_mace_energies.py \
        --batch_size $BATCH_SIZE \
        --gpu_idx $GPU_IDX \
        --model $MACE_MODEL
    echo "[Step 1] Done!"
    echo ""
else
    echo "[Step 1] Skipped"
fi

# Step 2: Process and save wrangler
if [ "$SKIP_STEP2" = false ]; then
    echo "[Step 2] Processing structures for CE..."
    python 2_process_save_wrangler_parallel.py \
        --n_workers $N_WORKERS
    echo "[Step 2] Done!"
    echo ""
else
    echo "[Step 2] Skipped"
fi

# Step 3: Fit ECIs
if [ "$SKIP_STEP3" = false ]; then
    echo "[Step 3] Fitting ECIs..."
    python 3_fit_ECIs.py \
        --two_step \
        --mu_point 1e-3 \
        --mu 1e-4
    echo "[Step 3] Done!"
    echo ""
else
    echo "[Step 3] Skipped"
fi

echo "============================================================"
echo "Pipeline Complete!"
echo "============================================================"
echo "Results saved in:"
echo "  - local_structures/"
echo "  - mace_energies/"
echo "  - ce_data/"
echo "============================================================"
