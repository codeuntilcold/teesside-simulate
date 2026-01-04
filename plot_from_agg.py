"""CLI for plotting from preprocessed aggregated data."""
from preprocess_seeds import load_agg, load_agg_final
from plotting_utils_v2 import SimulationPlotter
import numpy as np
import argparse
import os
import re


def detect_strategy_params(agg_dir, game_type, strategy, game_param):
    """Auto-detect available strategy params from files."""
    pattern = re.compile(rf'{game_type}_{strategy}_{game_param}_(pc|nc)=([^_]+)_')
    params = set()
    for f in os.listdir(agg_dir):
        match = pattern.match(f)
        if match:
            params.add(f"{match.group(1)}={match.group(2)}")
    return sorted(params, key=lambda x: float(x.split('=')[1]))


def load_agg_for_plotting(agg_dir, game_type, strategy, game_param, strategy_params):
    """Load all metrics for all strategy params into plotter-friendly format."""
    metrics = ['cooperator_frequency', 'cost', 'social_welfare']
    agg_data = {}

    for sp in strategy_params:
        agg_data[sp] = {}
        for metric in metrics:
            agg_data[sp][metric] = load_agg(agg_dir, game_type, strategy, game_param, sp, metric)

    return agg_data


def plot_timeseries(agg_dir, game_type, strategy, game_param, strategy_params, theta, output_file=None, show_std=True):
    """Load data and plot timeseries."""
    agg_data = load_agg_for_plotting(agg_dir, game_type, strategy, game_param, strategy_params)
    title = f'{game_type.upper()} {strategy.upper()} {game_param}, θ={theta}'

    plotter = SimulationPlotter()
    plotter.plot_timeseries_from_agg(agg_data, strategy_params, theta, title, show_std, output_file)


def build_heatmap_matrix(agg_dir, game_type, strategy, game_param, strategy_params, metric, population_size=10000):
    """Build heatmap data matrix from aggregated final values."""
    first_sp = strategy_params[0]
    first_data = load_agg_final(agg_dir, game_type, strategy, game_param, first_sp, metric)
    thetas = sorted(first_data.keys())

    data_matrix = np.zeros((len(thetas), len(strategy_params)))

    for sp_idx, sp in enumerate(strategy_params):
        final_data = load_agg_final(agg_dir, game_type, strategy, game_param, sp, metric)
        for theta_idx, theta in enumerate(thetas):
            if theta in final_data:
                val = final_data[theta]['mean']
                if metric == 'cooperator_frequency':
                    val = (val / population_size) * 100
                data_matrix[theta_idx, sp_idx] = val

    return data_matrix, thetas


def plot_heatmap(agg_dir, game_type, strategy, game_param, strategy_params, metric, output_file=None, population_size=10000):
    """Load data and plot heatmap."""
    data_matrix, thetas = build_heatmap_matrix(agg_dir, game_type, strategy, game_param, strategy_params, metric, population_size)
    title = f"{metric.replace('_', ' ').title()} - {game_type.upper()} {strategy.upper()} {game_param}"

    plotter = SimulationPlotter()
    plotter.plot_heatmap_from_agg(data_matrix, strategy_params, thetas, metric, title, output_file)


def build_efficiency_matrices(agg_dir, game_type, strategy, game_param, strategy_params):
    """Build cost and base welfare matrices for efficiency comparison."""
    # Get cost (summed over generations)
    cost_matrix, thetas = build_heatmap_matrix(
        agg_dir, game_type, strategy, game_param, strategy_params, 'cost', population_size=1
    )
    # Get social welfare (final value)
    welfare_matrix, _ = build_heatmap_matrix(
        agg_dir, game_type, strategy, game_param, strategy_params, 'social_welfare', population_size=1
    )

    # welfare_base = payoff = social_welfare + cost (at a=1)
    # For efficiency comparison: welfare_a = welfare_base - (cost / a)
    welfare_base = welfare_matrix + cost_matrix

    return {'cost': cost_matrix, 'welfare_base': welfare_base}, thetas

def build_diff_matrices(agg_dir, game_type, strategy, game_param, strategy_params):
    """Build cost, welfare_base and cooperator frequency matrices for diff line plot comparison."""
    # Get cost (final value)
    cost_matrix, thetas = build_heatmap_matrix(
        agg_dir, game_type, strategy, game_param, strategy_params, 'cost', population_size=1
    )
    # Get social welfare (final value)
    welfare_matrix, _ = build_heatmap_matrix(
        agg_dir, game_type, strategy, game_param, strategy_params, 'social_welfare', population_size=1
    )
    # Get cooperator frequency (final value)
    freq_matrix, _ = build_heatmap_matrix(
        agg_dir, game_type, strategy, game_param, strategy_params, 'cooperator_frequency'
    )

    # welfare_base = payoff = social_welfare + cost (at a=1)
    # For efficiency comparison: welfare_a = welfare_base - (cost / a)
    welfare_base = welfare_matrix + cost_matrix

    return {'cost': cost_matrix, 'welfare': welfare_base, 'coop_freq': freq_matrix}, thetas


def plot_efficiency_heatmap(agg_dir, game_type, strategy, game_param, strategy_params, a_values, output_file=None):
    """Load data and plot efficiency heatmap grid for different 'a' values."""
    data_matrices, thetas = build_efficiency_matrices(agg_dir, game_type, strategy, game_param, strategy_params)
    title = f"Efficiency Comparison - {game_type.upper()} {strategy.upper()} {game_param}"

    plotter = SimulationPlotter()
    plotter.plot_heatmap_grid_from_agg(data_matrices, strategy_params, thetas, a_values, title, output_file)

def plot_diff_line_chart(agg_dir, game_type, strategy, game_param, strategy_params, a_values, output_file=None):
    """Load data and plot diff line chart grid for different 'a' values."""
    data_matrices, thetas = build_diff_matrices(agg_dir, game_type, strategy, game_param, strategy_params)
    title = f"Diff Comparison - {game_type.upper()} {strategy.upper()} {game_param}"

    plotter = SimulationPlotter()
    plotter.plot_diff_plot_from_agg(data_matrices, strategy_params, thetas, a_values, title, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot from aggregated data.')
    parser.add_argument('--agg-dir', default='data_agg', help='Aggregated data directory')
    parser.add_argument('--game', required=True, choices=['pd', 'pgg'], help='Game type')
    parser.add_argument('--strategy', required=True, choices=['pop', 'neb'], help='Strategy')
    parser.add_argument('--game-param', required=True, help='e.g. r=3 or b=1.8')
    parser.add_argument('--plot-type', required=True, choices=['timeseries', 'heatmap', 'efficiency', 'diff'], help='Plot type')
    parser.add_argument('--theta', type=float, help='Theta value (for timeseries)')
    parser.add_argument('--metric', default='cooperator_frequency', help='Metric (for heatmap)')
    parser.add_argument('--a-values', type=float, nargs='+', default=[0.5, 1, 1.5], help='Efficiency values (for efficiency plot)')
    parser.add_argument('--output', help='Output file')
    parser.add_argument('--fig-prefix', default='', help='Optional subdirectory under fig/ (e.g. "cpp" -> fig/cpp/)')
    args = parser.parse_args()

    # Auto-detect strategy params from files
    all_params = detect_strategy_params(args.agg_dir, args.game, args.strategy, args.game_param)
    if not all_params:
        print(f"Error: No data files found for {args.game}_{args.strategy}_{args.game_param}")
        exit(1)

    # Filter params based on plot type (POP only)
    if args.strategy == 'pop':
        if args.plot_type == 'timeseries' or args.plot_type == 'diff':
            # Timeseries: only 0.25, 0.5, 0.75, 1.0
            allowed = {0.25, 0.5, 0.75, 1.0}
            strategy_params = [p for p in all_params if float(p.split('=')[1]) in allowed]
        else:
            # Heatmap: only >= 0.9
            strategy_params = [p for p in all_params if float(p.split('=')[1]) >= 0.9]
        # Deduplicate by float value (e.g., pc=1 and pc=1.0 are same)
        seen = set()
        deduped = []
        for p in strategy_params:
            val = float(p.split('=')[1])
            if val not in seen:
                seen.add(val)
                deduped.append(p)
        strategy_params = deduped
    else:
        strategy_params = all_params

    print(f"Using strategy params: {strategy_params}")

    # Auto-generate output filename if not provided
    output_dir = f"fig/{args.fig_prefix}" if args.fig_prefix else 'fig'
    os.makedirs(output_dir, exist_ok=True)

    if args.plot_type == 'timeseries':
        if not args.theta:
            print("Error: --theta required for timeseries")
            exit(1)
        output_file = args.output or f"{output_dir}/{args.game}_{args.strategy}_{args.game_param}_timeseries_theta={args.theta}.png"
        plot_timeseries(args.agg_dir, args.game, args.strategy, args.game_param, strategy_params, args.theta, output_file)
    elif args.plot_type == 'heatmap':
        output_file = args.output or f"{output_dir}/{args.game}_{args.strategy}_{args.game_param}_heatmap_{args.metric}.png"
        plot_heatmap(args.agg_dir, args.game, args.strategy, args.game_param, strategy_params, args.metric, output_file)
    elif args.plot_type == 'efficiency':
        a_str = '_'.join(str(a) for a in args.a_values)
        output_file = args.output or f"{output_dir}/{args.game}_{args.strategy}_{args.game_param}_efficiency_a={a_str}.png"
        plot_efficiency_heatmap(args.agg_dir, args.game, args.strategy, args.game_param, strategy_params, args.a_values, output_file)
    elif args.plot_type == 'diff':
        a_str = '_'.join(str(a) for a in args.a_values)
        output_file = args.output or f"{output_dir}/{args.game}_{args.strategy}_{args.game_param}_diff_a={a_str}.png"
        plot_diff_line_chart(args.agg_dir, args.game, args.strategy, args.game_param, strategy_params, args.a_values, output_file)
