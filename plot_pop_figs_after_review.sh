#!/bin/bash

source .venv/bin/activate

PC_4=(0.25 0.5 0.75 1.0)
PC_ALL=(0.25 0.5 0.75 0.9 0.92 0.94 0.96 0.98 1.0)

heatmap_pop() {
  local agg_dir=$1
  local fig_prefix=$2
  local game=$3
  local game_param=$4
  shift 4
  python plot_from_agg.py \
    --agg-dir "$agg_dir" --fig-prefix "$fig_prefix" \
    --game "$game" --strategy pop --game-param "$game_param" \
    --plot-type efficiency --pc-values "$@" &
}

for b in 1.2 1.8 2.0; do
  heatmap_pop data_agg_cpp            cpp/pc4            pd  "b=$b" "${PC_4[@]}"
  heatmap_pop data_agg_det_go         det/pc4            pd  "b=$b" "${PC_4[@]}"
done
for r in 1.5 3.0 4.5; do
  heatmap_pop data_agg_go_5groups     5groups/nondet/pc4 pgg "r=$r" "${PC_4[@]}"
  heatmap_pop data_agg_det_go_5groups 5groups/det/pc4    pgg "r=$r" "${PC_4[@]}"
done

for b in 1.2 1.8 2.0; do
  heatmap_pop data_agg_cpp            cpp/appendix            pd  "b=$b" "${PC_ALL[@]}"
  heatmap_pop data_agg_det_go         det/appendix            pd  "b=$b" "${PC_ALL[@]}"
done
for r in 1.5 3.0 4.5; do
  heatmap_pop data_agg_go_5groups     5groups/nondet/appendix pgg "r=$r" "${PC_ALL[@]}"
  heatmap_pop data_agg_det_go_5groups 5groups/det/appendix    pgg "r=$r" "${PC_ALL[@]}"
done

python plot_optimal_theta.py --agg-dir data_agg_det_go --output fig/det/optimal_theta_summary.png &
python plot_optimal_theta.py --agg-dir data_agg_det_go --pd-params b=1.2 b=1.8 b=2.0 --output fig/det/appendix/optimal_theta_summary_with_b2.png &

wait
echo "Done."
