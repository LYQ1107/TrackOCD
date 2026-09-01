#!/usr/bin/env bash
set -u
cd /data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
export PYTHONPATH=/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
P=/home/lwr/anaconda3/envs/ocd_ovmot_discovery/bin/python

CUDA_VISIBLE_DEVICES=4 $P -u src/orbit_msr/train.py --variant KG1 --balanced \
  --epochs 40 --episodes_per_epoch 12 --output_dir msr_kg1 > runs/orbit_msr/train_kg1.log 2>&1 &
p1=$!
CUDA_VISIBLE_DEVICES=5 $P -u src/orbit_msr/train.py --variant KG2 --balanced --margin \
  --epochs 40 --episodes_per_epoch 12 --output_dir msr_kg2 > runs/orbit_msr/train_kg2.log 2>&1 &
p2=$!
CUDA_VISIBLE_DEVICES=8 $P -u src/orbit_msr/train.py --variant NR1 --balanced --update_radius \
  --epochs 40 --episodes_per_epoch 12 --output_dir msr_nr1 > runs/orbit_msr/train_nr1.log 2>&1 &
p3=$!
CUDA_VISIBLE_DEVICES=9 $P -u src/orbit_msr/train.py --variant NR2 --balanced --update_radius \
  --mem_scale_norm --epochs 40 --episodes_per_epoch 12 --output_dir msr_nr2 > runs/orbit_msr/train_nr2.log 2>&1 &
p4=$!
echo "launched $p1 $p2 $p3 $p4"
wait $p1; r1=$?
wait $p2; r2=$?
wait $p3; r3=$?
wait $p4; r4=$?
echo "batch_syn=$r1,$r2,$r3,$r4"
test $r1 -eq 0 -a $r2 -eq 0 -a $r3 -eq 0 -a $r4 -eq 0
