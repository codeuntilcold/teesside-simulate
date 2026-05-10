"""Plot θ* (theta for optimal social welfare and cost) summary across all experiments."""
import argparse
import os
from typing import List
import numpy as np
from plot_from_agg import detect_strategy_params, build_metric_matrices
from plotting_utils_v2 import plot_optimal_theta_summary, MetricMatrices, OptimalRecord

DEFAULT_PD_PARAMS = ['b=1.2', 'b=1.8', 'b=2.0']
DEFAULT_PGG_PARAMS = ['r=1.5', 'r=3.0', 'r=4.5']
STRATEGIES = ['pop', 'neb']
COOP_THRESHOLD = 90.0


def find_optimal_thetas_per_a(data_matrices: MetricMatrices, thetas, a_values):
    """
    For each a value and each strategy_param column, find:
      - SW: theta that maximizes welfare_a, regardless of coop level
      - Cost: lowest theta where coop > 90%

    Returns: {
        'sw': {a: [theta_stars]},
        'cost': {a: [{'theta': float, 'feasible': bool, 'max_coop': float}]}
    }
    """
    sw_result = {}
    cost_result = {}
    thetas_arr = np.array(thetas)
    coop_freq = data_matrices['coop_freq']  # shape [n_thetas, n_params]

    for a in a_values:
        welfare_a = data_matrices['welfare'] + (a - 1) * data_matrices['cost']

        sw_result[a] = thetas_arr[np.argmax(welfare_a, axis=0)].tolist()
        cost_result[a] = []
        n_params = coop_freq.shape[1]
        for sp_idx in range(n_params):
            coop_col = coop_freq[:, sp_idx]
            feasible_mask = coop_col > COOP_THRESHOLD

            if feasible_mask.any():
                feasible_idx = np.where(feasible_mask)[0]
                cost_result[a].append({
                    'theta': thetas_arr[feasible_idx[0]],
                    'feasible': True,
                    'max_coop': coop_col[feasible_idx[0]],
                })
            else:
                best_coop_idx = int(np.argmax(coop_col))
                cost_result[a].append({
                    'theta': thetas_arr[best_coop_idx],
                    'feasible': False,
                    'max_coop': coop_col[best_coop_idx],
                })

    return {'sw': sw_result, 'cost': cost_result}


def build_all_optimal_data(agg_dir, a_values, games_config) -> List[OptimalRecord]:
    """Build θ* for SW and cost for all (a, game, strategy, game_param, sp) cells."""
    records: List[OptimalRecord] = []

    for game, params in games_config.items():
        for strategy in STRATEGIES:
            for game_param in params:
                all_params = detect_strategy_params(agg_dir, game, strategy, game_param)
                if not all_params:
                    continue

                # Filter: POP pc >= 0.9, NEB nc >= 3
                if strategy == 'pop':
                    all_params = [p for p in all_params if float(p.split('=')[1]) >= 0.9]
                elif strategy == 'neb':
                    all_params = [p for p in all_params if float(p.split('=')[1]) >= 3]
                if not all_params:
                    continue

                data_matrices, thetas = build_metric_matrices(
                    agg_dir, game, strategy, game_param, all_params
                )
                results = find_optimal_thetas_per_a(data_matrices, thetas, a_values)

                for a in a_values:
                    for sp_idx, sp in enumerate(all_params):
                        cost_info = results['cost'][a][sp_idx]
                        records.append(OptimalRecord(
                            a=a, game=game, strategy=strategy,
                            game_param=game_param, sp=sp,
                            theta_sw=results['sw'][a][sp_idx],
                            theta_cost=cost_info['theta'],
                            feasible=cost_info['feasible'],
                            max_coop=cost_info['max_coop'],
                        ))

    return records


def main():
    parser = argparse.ArgumentParser(description='Plot θ* summary for all experiments.')
    parser.add_argument('--agg-dir', default='data_agg_det_go', help='Aggregated data directory')
    parser.add_argument('--a-values', type=float, nargs='+', default=[0.5, 1.0, 1.5],
                        help='Efficiency a values')
    parser.add_argument('--pd-params', nargs='+', default=DEFAULT_PD_PARAMS,
                        help='PD game_params to include (default: main-text view, no b=2.0)')
    parser.add_argument('--pgg-params', nargs='+', default=DEFAULT_PGG_PARAMS,
                        help='PGG game_params to include')
    parser.add_argument('--output', default='fig/det/optimal_theta_summary.png', help='Output file')
    parser.add_argument('--show', action='store_true', help='Show plot')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    games_config = {'pd': args.pd_params, 'pgg': args.pgg_params}
    print(f"Building optimal θ data for: {games_config}")
    records = build_all_optimal_data(args.agg_dir, args.a_values, games_config)

    print("\nGenerating plot...")
    plot_optimal_theta_summary(
        records=records,
        a_values=args.a_values,
        title='θ* for Optimal Social Welfare and Cost',
        output_filename=args.output,
        show_plot=args.show
    )


if __name__ == "__main__":
    main()
