#!/usr/bin/env bash

set -euo pipefail

go build -o teesside-go main.go

t=$(( $(nproc) - 1 ))
parallel_params="--halt now,fail=1 -j $t --eta --bar"

k_values="0.1 0.3 0.5"
pd="--game-type pd --beta 1.8 --theta-start 0 --theta-end 10 --theta-step 0.1"
pgg="--game-type pgg --r 3.0 --theta-start 0 --theta-end 10 --theta-step 0.1"

# PD b=1.8
parallel $parallel_params ./teesside-go $pd --strategy pop --pc 1.0 \
    --fermi-k {1} --seed-start {2} --seed-end '$(('{2}'+1))' \
    --output-dir 'data_knoise/pop_b=1.8_k{1}' \
    ::: $k_values ::: {0..49}

parallel $parallel_params ./teesside-go $pd --strategy neb --nc {3} \
    --fermi-k {1} --seed-start {2} --seed-end '$(('{2}'+1))' \
    --output-dir 'data_knoise/neb_b=1.8_k{1}' \
    ::: $k_values ::: {0..49} ::: 3 4

# PGG r=3.0
parallel $parallel_params ./teesside-go $pgg --strategy pop --pc 1.0 \
    --fermi-k {1} --seed-start {2} --seed-end '$(('{2}'+1))' \
    --output-dir 'data_knoise/pop_pgg_r=3.0_k{1}' \
    ::: $k_values ::: {0..49}

# NEB, n_C = 3
parallel $parallel_params ./teesside-go $pgg --strategy neb --nc {3} \
    --fermi-k {1} --seed-start {2} --seed-end '$(('{2}'+1))' \
    --output-dir 'data_knoise/neb_pgg_r=3.0_k{1}' \
    ::: $k_values ::: {0..49} ::: 3 4
