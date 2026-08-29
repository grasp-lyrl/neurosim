set -e
for ds in 1 4 8; do
  /opt/conda/envs/neurosim/bin/python applications/rl/train_velocity_dodge_bc.py \
    --experiment-config applications/rl/configs/velocity_dodge_privileged_baseline.yaml \
    --dataset outputs/rl/bc/dagger/dagger_iter2.h5 \
    --output outputs/rl/bc/sweep/clone_ds${ds}.json \
    --epochs 25 --samples-per-epoch 6000 --batch-size 64 \
    --blank-previous-action --downsample-events ${ds} --gated-onset-window 4 \
    --holdout-fraction 0.25 --holdout-seed 0 \
    --eval-episodes 20 --eval-every 12 --eval-seed0 9001 \
    --device cuda:0 > outputs/rl/bc/sweep/ds${ds}.log 2>&1
  echo "=== ds=${ds} done ==="
done
