# declare -a strideList=(0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0 1.05 1.1 1.15 1.2 1.25 1.3 1.35 1.4 1.45 1.5 1.55 1.6 1.65 1.7 1.75 1.8 1.85 1.9 1.95 2)
declare -a strideList=(1.1 1.15 1.2 1.25 1.3 1.35 1.4 1.45 1.5 1.55 1.6 1.65 1.7 1.75 1.8 1.85 1.9 1.95 2)
declare -a bucketList=(1 2 3 4 5 6 7 8 9 10)
# declare -a bucketList=(2 3)

OUTDIR='output/eval_heuristic'

for i in "${strideList[@]}"
do
    for j in "${bucketList[@]}"
    do
        python train/train_net_det.py --cfg cfgs/det_sample_all.yaml OUTPUT_DIR $OUTDIR HEURISTIC_STRIDE $i HEURISTIC_BUCKETS $j
    done
   # do whatever on $i
done