#!/bin/bash
# Fill the 200k->500k gap with 300k/400k checkpoints (chained 100k at a time)
# and probe them; 500k' must equal the existing single-run 500k checkpoint.
set -euo pipefail
ds=$1
w=explore/nse_alt/work/$ds
g=$w/gap
mkdir -p $g
prev=$w/ck/ck_200000.h5
for s in 300000 400000 500000; do
  ./build-min/coast -n 100000 --load-state $prev --save-state $g/ck_$s.h5 $w/cfg/train.json > $g/train_$s.log 2>&1
  prev=$g/ck_$s.h5
  for c in real perm unif; do
    rm -f $g/${c}_$s.h5
    ./build-min/coast -n 2000 --load-state $prev --log-buffers $g/${c}_$s.h5 $w/cfg/$c.json > $g/probe_${c}_$s.log 2>&1
  done
done
echo "gapfill $ds done"
