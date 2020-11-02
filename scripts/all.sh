#!/bin/bash
# cd ops
# bash clean.sh
# bash make.sh
# cd ..

set -x
set -e

export CUDA_VISIBLE_DEVICES=0

python kitti/prepare_data.py --gen_train --gen_val --gen_val_rgb_detection

OUTDIR='output/all_train'
python train/train_net_det.py --cfg cfgs/det_sample_all.yaml OUTPUT_DIR $OUTDIR
python train/test_net_det.py --cfg cfgs/det_sample_all.yaml OUTPUT_DIR $OUTDIR TEST.WEIGHTS $OUTDIR/model_best.pth 

# python kitti/prepare_data_refine.py --gen_train --gen_val_det --gen_val_rgb_detection

# OUTDIR='output/all_train_refine'
# python train/train_net_det.py --cfg cfgs/refine_car.yaml OUTPUT_DIR $OUTDIR 
# python train/test_net_det.py --cfg cfgs/refine_car.yaml OUTPUT_DIR $OUTDIR TEST.WEIGHTS $OUTDIR/model_best.pth