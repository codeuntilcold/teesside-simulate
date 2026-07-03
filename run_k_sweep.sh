#!/usr/bin/env bash

go build -o teesside-go main.go

t=$(( $(nproc) - 1 ))
parallel_params="--halt now,fail=1 -j $t --eta --bar"

common="--game-type pd --beta 1.8 --theta-start 0 --theta-end 10 --theta-step 0.1"
k_values="0.1 0.2 0.3 0.5 1.0"

# POP, p_C = 1.0 -- one output dir per K (dir name must start with pop_ for preprocess)
parallel $parallel_params ./teesside-go $common --strategy pop --pc 1.0 \
    --fermi-k {1} --seed-start {2} --seed-end '$(('{2}'+1))' \
    --output-dir data_ksweep/pop_k{1} \
    ::: $k_values \
    ::: {0..49}

# NEB, n_C = 3
parallel $parallel_params ./teesside-go $common --strategy neb --nc 3 \
    --fermi-k {1} --seed-start {2} --seed-end '$(('{2}'+1))' \
    --output-dir data_ksweep/neb_k{1} \
    ::: $k_values \
    ::: {0..49}

# for d in data_ksweep/*_k*; do
#     python preprocess_seeds.py --data-dir "$d" --output-dir "${d/data_ksweep/data_ksweep_agg}"
# done
