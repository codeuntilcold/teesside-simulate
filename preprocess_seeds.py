"""Preprocess seed data into aggregated mean/std files for faster plotting."""
import argparse
import numpy as np
import os
import csv
import re
import ast
from typing import Any, Dict, List


class DataLoader:
    """Handles loading and parsing per-seed simulation CSVs."""

    METRIC_COLUMNS = {
        'cooperator_frequency': 'Cooperator_Frequency',
        'cost': 'Cost',
        'social_welfare': 'Social_Welfare',
        'population_payoff': 'Payoff',
    }

    @staticmethod
    def parse_array(array_str: str) -> List[float]:
        """Parse array string - handles np.int64(val), [1,2,3], and space-separated formats."""
        pattern = r'np\.(int64|float64)\(([^)]+)\)'
        matches = re.findall(pattern, array_str)
        if matches:
            return [float(val) for _, val in matches]

        array_str = array_str.strip().strip('[]')

        if ',' in array_str:
            return [float(x.strip()) for x in array_str.split(',') if x.strip()]

        return [float(x) for x in array_str.split() if x.strip()]

    @staticmethod
    def load_csv(filepath: str) -> List[Dict[str, Any]]:
        with open(filepath, 'r') as f:
            return list(csv.DictReader(f))

    @classmethod
    def load_metric_data(cls, data_dir: str, metric: str, param_value: str, seed: int,
                         param_name: str = 'pc', theta_range: str = None) -> List[Dict[str, Any]]:
        """Load metric data with flexible param_name (pc or nc). Auto-detects _det suffix."""
        if theta_range:
            filepath = f"{data_dir}/seed_{seed}_theta_{theta_range}_{param_name}={param_value}_{metric}.csv"
            if not os.path.exists(filepath):
                filepath = f"{data_dir}/seed_{seed}_theta_{theta_range}_{param_name}={param_value}_det_{metric}.csv"
        else:
            filepath = f"{data_dir}/seed_{seed}_{param_name}={param_value}_{metric}.csv"
            if not os.path.exists(filepath):
                filepath = f"{data_dir}/seed_{seed}_{param_name}={param_value}_det_{metric}.csv"
        return cls.load_csv(filepath)

    @classmethod
    def parse_metric_from_row(cls, row: Dict[str, Any], metric: str) -> List[float]:
        """Parse metric array from row."""
        column = cls.METRIC_COLUMNS.get(metric)
        if not column:
            raise ValueError(f"Unknown metric: {metric}")
        return cls.parse_array(row[column])


def load_agg(agg_dir, game_type, strategy, game_param, strategy_param, metric):
    """Load preprocessed aggregated data.

    Args:
        agg_dir: Directory containing aggregated files
        game_type: 'pd' or 'pgg'
        strategy: 'pop' or 'neb'
        game_param: e.g. 'r=3' or 'b=1.8'
        strategy_param: e.g. 'pc=0.25' or 'nc=1'
        metric: 'cooperator_frequency', 'cost', or 'social_welfare'

    Returns:
        dict: {theta: {'mean': [...], 'std': [...]}}
    """
    filepath = f"{agg_dir}/{game_type}_{strategy}_{game_param}_{strategy_param}_{metric}.csv"
    result = {}

    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            theta = float(row['Theta'])
            result[theta] = {
                'mean': ast.literal_eval(row['Mean_Timeseries']),
                'std': ast.literal_eval(row['Std_Timeseries'])
            }

    return result


def load_agg_final(agg_dir, game_type, strategy, game_param, strategy_param, metric):
    """Load just the final values (last element of timeseries).

    Returns:
        dict: {theta: {'mean': float, 'std': float}}
    """
    data = load_agg(agg_dir, game_type, strategy, game_param, strategy_param, metric)
    return {
        theta: {'mean': v['mean'][-1], 'std': v['std'][-1]}
        for theta, v in data.items()
    }


def load_agg_total(agg_dir, game_type, strategy, game_param, strategy_param, metric):
    """Load sum of timeseries values (total across all generations).

    Returns:
        dict: {theta: {'mean': float, 'std': float}}
    """
    data = load_agg(agg_dir, game_type, strategy, game_param, strategy_param, metric)
    return {
        theta: {'mean': sum(v['mean']), 'std': sum(v['std'])}
        for theta, v in data.items()
    }


def parse_path_metadata(data_dir):
    """Extract game_type, strategy, and game params from directory path."""
    path = data_dir.rstrip('/')
    dirname = os.path.basename(path)

    # Detect strategy (pop or neb)
    if dirname.startswith('pop_'):
        strategy = 'pop'
    elif dirname.startswith('neb_'):
        strategy = 'neb'
    else:
        strategy = 'unknown'

    # Detect game type and params (default to PD if no PGG found)
    if '_pgg_' in dirname or 'pgg_r=' in dirname:
        game_type = 'pgg'
        r_match = re.search(r'r=([0-9.]+)', dirname)
        game_param = f"r={r_match.group(1)}" if r_match else ""
    else:
        game_type = 'pd'
        b_match = re.search(r'b=([0-9.]+)', dirname)
        game_param = f"b={b_match.group(1)}" if b_match else "b=1.8"

    return {
        'game_type': game_type,
        'strategy': strategy,
        'game_param': game_param
    }


def detect_data_format(data_dir):
    """Auto-detect seeds, param_name, param_values, and theta_range from filenames."""
    seeds = set()
    param_values = set()
    theta_range = None
    param_name = None

    seed_pattern = re.compile(r'seed_(\d+)_')
    theta_pattern = re.compile(r'theta_([0-9.]+-[0-9.]+)_')
    pc_pattern = re.compile(r'pc=([^_]+)_')
    nc_pattern = re.compile(r'nc=([^_]+)_')

    for filename in os.listdir(data_dir):
        seed_match = seed_pattern.match(filename)
        if seed_match:
            seeds.add(int(seed_match.group(1)))

        theta_match = theta_pattern.search(filename)
        if theta_match:
            theta_range = theta_match.group(1)

        pc_match = pc_pattern.search(filename)
        if pc_match:
            param_name = 'pc'
            param_values.add(pc_match.group(1))

        nc_match = nc_pattern.search(filename)
        if nc_match:
            param_name = 'nc'
            param_values.add(nc_match.group(1))

    return sorted(seeds), param_name, sorted(param_values), theta_range


def preprocess_data(data_dir, output_dir):
    """Aggregate seed data into mean/std files."""
    loader = DataLoader()
    metrics = ['cooperator_frequency', 'cost', 'social_welfare']

    metadata = parse_path_metadata(data_dir)
    seeds, param_name, param_values, theta_range = detect_data_format(data_dir)

    print(f"Data dir: {data_dir}")
    print(f"Game: {metadata['game_type']}, Strategy: {metadata['strategy']}, {metadata['game_param']}")
    print(f"Detected {len(seeds)} seeds: {seeds[0]}-{seeds[-1]}")
    print(f"Detected {param_name}={param_values}")
    print(f"Detected theta range: {theta_range}")

    os.makedirs(output_dir, exist_ok=True)

    game_type = metadata['game_type']
    strategy = metadata['strategy']
    game_param = metadata['game_param']

    for param_value in param_values:
        print(f"\nProcessing {param_name}={param_value}...")
        strategy_param = f"{param_name}={param_value}"

        for metric in metrics:
            all_timeseries = {}

            for seed in seeds:
                try:
                    metric_data = loader.load_metric_data(
                        data_dir, metric, param_value, seed,
                        param_name=param_name, theta_range=theta_range
                    )

                    for row in metric_data:
                        theta = round(float(row['Theta']), 1)
                        values = loader.parse_metric_from_row(row, metric)

                        if theta not in all_timeseries:
                            all_timeseries[theta] = []

                        if values:
                            all_timeseries[theta].append(values)

                except FileNotFoundError:
                    continue

            if not all_timeseries:
                print(f"  No data for {metric}")
                continue

            # Cost: runs that converged early pay no further cost, so pad with 0
            # to avoid biasing load_agg_total upward. Other metrics flatline at
            # their converged value.
            pad_value = 0 if metric == 'cost' else None
            ts_file = f"{output_dir}/{game_type}_{strategy}_{game_param}_{strategy_param}_{metric}.csv"
            with open(ts_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Theta', 'Mean_Timeseries', 'Std_Timeseries'])
                for theta in sorted(all_timeseries.keys()):
                    series_list = all_timeseries[theta]
                    if not series_list:
                        continue
                    max_len = max(len(s) for s in series_list)
                    padded = [
                        list(s) + [pad_value if pad_value is not None else s[-1]] * (max_len - len(s))
                        for s in series_list
                    ]
                    arr = np.array(padded)
                    writer.writerow([theta, np.mean(arr, axis=0).tolist(), np.std(arr, axis=0).tolist()])

        print(f"  Done")


def main():
    parser = argparse.ArgumentParser(description='Preprocess seed data into mean/std files.')
    parser.add_argument('--data-dir', required=True, help='Input data directory')
    parser.add_argument('--output-dir', required=True, help='Output directory for aggregated files')
    args = parser.parse_args()

    preprocess_data(args.data_dir, args.output_dir)
    print(f"\nAggregated files written to: {args.output_dir}")


if __name__ == "__main__":
    main()
