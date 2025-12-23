import matplotlib.pyplot as plt
import numpy as np
import csv
import re
import os
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class PlotConfig:
    COLORS = {
        'cooperator_frequency': '#2E86AB',
        'cost': '#A23B72',
        'social_welfare': '#F18F01',
        'population_payoff': '#06A77D'
    }

    IJCAI_COLORS = {
        'cooperator': '#6B7FD7',    # Blue-violet
        'defector': '#FF7875',       # Coral-red
        'cost': '#B88BBD',           # Purple-mauve
        'welfare': '#F18F01'         # Orange
    }

    max_generations: int = 50
    default_population_size: int = 10000

    # Plot styling
    grid_alpha: float = 0.3
    linewidth: float = 2
    fill_alpha: float = 0.3
    dpi: int = 300

    # Font sizes
    title_fontsize: int = 16
    subtitle_fontsize: int = 13
    label_fontsize: int = 12
    tick_fontsize: int = 10
    annotation_fontsize: int = 8

    # Figure sizes
    single_plot_figsize: Tuple[int, int] = (12, 13)
    grid_plot_figsize: Tuple[int, int] = (20, 16)
    heatmap_figsize: Tuple[int, int] = (14, 8)


class DataLoader:
    METRIC_COLUMN_MAP = {
        'population_payoff': 'Payoff',
        'cooperator_frequency': 'Cooperator_Frequency',
        'cost': 'Cost',
        'social_welfare': 'Social_Welfare'
    }

    @staticmethod
    def parse_numpy_array(array_str: str) -> List[float]:
        """Parse numpy array string in wrapped format: np.int64(value)"""
        pattern = r'np\.(int64|float64)\(([^)]+)\)'
        matches = re.findall(pattern, array_str)
        return [float(val) for _, val in matches]

    @staticmethod
    def parse_plain_array(array_str: str) -> List[float]:
        """Parse numpy array string in plain format: [value1 value2 ...]"""
        # Remove brackets and split by whitespace
        array_str = array_str.strip()
        if array_str.startswith('['):
            array_str = array_str[1:]
        if array_str.endswith(']'):
            array_str = array_str[:-1]

        # Split and convert to floats, filtering out empty strings
        values = [float(x) for x in array_str.split() if x.strip()]
        return values

    @staticmethod
    def parse_metric_from_row(row: Dict[str, Any], metric: str) -> List[float]:
        column = DataLoader.METRIC_COLUMN_MAP.get(metric)
        if column is None:
            raise ValueError(f"Unknown metric: {metric}")
        return DataLoader.parse_numpy_array(row[column])

    @staticmethod
    def load_csv_data(filepath: str) -> List[Dict[str, Any]]:
        data = []
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        return data

    @staticmethod
    def load_metric_data(data_dir: str, metric: str, pc_value: str, seed: int) -> List[Dict[str, Any]]:
        filepath = f"{data_dir}/seed_{seed}_pc={pc_value}_{metric}.csv"
        return DataLoader.load_csv_data(filepath)

    @staticmethod
    def get_unique_theta_values(data_dir: str, pc_value: str, seed: int) -> List[float]:
        data = DataLoader.load_metric_data(data_dir, 'cooperator_frequency', pc_value, seed)
        return sorted(set(float(row['Theta']) for row in data))

    @staticmethod
    def load_neb_data(data_dir: str, nc_value: int) -> List[Dict[str, Any]]:
        """
        Load NEB strategy data from consolidated file format.

        Args:
            data_dir: Directory containing NEB data files
            nc_value: Neighborhood cooperator threshold value (1, 2, 3, or 4)

        Returns:
            List of dictionaries containing NEB data
        """
        filepath = f"{data_dir}/nc_{nc_value}_final_all.csv"
        return DataLoader.load_csv_data(filepath)

    @staticmethod
    def parse_neb_metric(row: Dict[str, Any], metric: str) -> List[float]:
        """
        Parse metric data from NEB data row.

        Args:
            row: Dictionary containing NEB CSV row data
            metric: Metric name ('cooperator_frequency', 'cost', 'social_welfare', 'population_payoff')

        Returns:
            Parsed metric data as list of floats
        """
        column_map = {
            'cooperator_frequency': 'Final_Cooperator_Frequency',
            'cost': 'Final_Cost',
            'social_welfare': 'Final_Social_Welfare',
            'population_payoff': 'Final_Fitnesses'
        }

        column = column_map.get(metric)
        if column is None:
            raise ValueError(f"Unknown NEB metric: {metric}")

        return DataLoader.parse_plain_array(row[column])


class SimulationPlotter:
    def __init__(self, config: Optional[PlotConfig] = None):
        self.config = config or PlotConfig()
        self.loader = DataLoader()

    def _plot_single_metric_heatmap(
        self,
        ax,
        data: np.ndarray,
        pc_values: List[str],
        theta_values: List[float],
        metric: str,
        title: str,
        colorbar_label: str = 'Value',
        cmap: str = 'plasma',
        annotation_fmt: Optional[str] = None,
        annotation_fontsize: int = 8
    ):
        # Create heatmap
        im = ax.imshow(data, aspect='auto', cmap=cmap, origin='lower')

        # Set ticks and labels
        ax.set_xticks(np.arange(len(pc_values)))
        ax.set_xticklabels([f'{float(pc):.2f}' for pc in pc_values])
        ax.set_yticks(np.arange(len(theta_values)))
        ax.set_yticklabels([f'{t:.1f}' for t in theta_values])
        ax.set_xlabel('$p_C / Z$', fontsize=self.config.label_fontsize)
        
        ax.set_ylabel('$\\theta$', fontsize=self.config.label_fontsize)
        ax.set_title(title, fontsize=self.config.subtitle_fontsize, fontweight='bold')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(colorbar_label, rotation=270, labelpad=15)

        # Add text annotations
        if annotation_fmt is None:
            # Auto-detect format: 1 decimal for percentages, 0 for other metrics
            annotation_fmt = '.1f' if metric == 'cooperator_frequency' else '.0f'

        for i in range(len(theta_values)):
            for j in range(len(pc_values)):
                text = ax.text(j, i, f'{data[i, j]:{annotation_fmt}}',
                              ha="center", va="center",
                              color="white" if data[i, j] < data.max() * 0.5 else "black",
                              fontsize=annotation_fontsize)

    def plot_ijcai_figure1_grid(
        self,
        data_dir: str,
        pc_values: List[str],
        theta_value: float,
        seed: int = 0,
        population_size: int = 10000,
        output_filename: Optional[str] = None,
        show_plot: bool = False
    ) -> str:
        n_cols = len(pc_values)

        # IJCAI paper colors
        C_COLOR = self.config.IJCAI_COLORS['cooperator']
        D_COLOR = self.config.IJCAI_COLORS['defector']
        COST_COLOR = self.config.IJCAI_COLORS['cost']
        WELFARE_COLOR = self.config.IJCAI_COLORS['welfare']

        # Create figure: 3 rows × n_cols columns
        fig, axes = plt.subplots(3, n_cols, figsize=(4*n_cols, 6),
                                gridspec_kw={'hspace': 0.3, 'wspace': 0.25})

        # Ensure axes is 2D even with single column
        if n_cols == 1:
            axes = axes.reshape(-1, 1)

        for col_idx, pc_value in enumerate(pc_values):
            # Load data
            coop_data = self.loader.load_metric_data(data_dir, 'cooperator_frequency', pc_value, seed)
            cost_data = self.loader.load_metric_data(data_dir, 'cost', pc_value, seed)
            welfare_data = self.loader.load_metric_data(data_dir, 'social_welfare', pc_value, seed)

            # Find the row for this theta value
            coop_row = next((r for r in coop_data if abs(float(r['Theta']) - theta_value) < 0.01), None)
            cost_row = next((r for r in cost_data if abs(float(r['Theta']) - theta_value) < 0.01), None)
            welfare_row = next((r for r in welfare_data if abs(float(r['Theta']) - theta_value) < 0.01), None)

            if not coop_row or not cost_row or not welfare_row:
                print(f"Warning: No data for pc={pc_value}, theta={theta_value}")
                continue

            # Parse data
            coop_counts = self.loader.parse_numpy_array(coop_row['Cooperator_Frequency'])
            costs = self.loader.parse_numpy_array(cost_row['Cost'])
            welfare = self.loader.parse_numpy_array(welfare_row['Social_Welfare'])

            # Find convergence point (where welfare becomes 0 after being non-zero)
            convergence_idx = None
            for i in range(len(welfare)):
                if i > 0 and welfare[i] == 0 and welfare[i-1] > 0:
                    convergence_idx = i
                    break

            # Find full cooperation point (when cooperation reaches ~100%)
            # Use 99.9% threshold since data might not hit exactly 10000
            full_coop_idx = None
            for i in range(len(coop_counts)):
                if coop_counts[i] >= population_size * 0.999:
                    full_coop_idx = i
                    break

            # If converged, pad with last non-zero value
            if convergence_idx is not None:
                last_valid_welfare = welfare[convergence_idx - 1]
                welfare = welfare[:convergence_idx] + [last_valid_welfare] * (self.config.max_generations - convergence_idx)

                # Also ensure coop and cost are extended to max_generations
                if len(coop_counts) < self.config.max_generations:
                    coop_counts = coop_counts + [coop_counts[-1]] * (self.config.max_generations - len(coop_counts))
                if len(costs) < self.config.max_generations:
                    costs = costs + [costs[-1]] * (self.config.max_generations - len(costs))
            else:
                # Normal padding if needed
                actual_len = min(len(coop_counts), len(costs), len(welfare))
                if actual_len < self.config.max_generations:
                    coop_counts = coop_counts[:actual_len] + [coop_counts[actual_len-1]] * (self.config.max_generations - actual_len)
                    costs = costs[:actual_len] + [costs[actual_len-1]] * (self.config.max_generations - actual_len)
                    welfare = welfare[:actual_len] + [welfare[actual_len-1]] * (self.config.max_generations - actual_len)

            # For pc=1.0, cost should be 0 after full cooperation is reached
            if pc_value == "1.0" and full_coop_idx is not None:
                # Set cost to 0 from the point of full cooperation onward
                costs = costs[:full_coop_idx + 1] + [0.0] * (self.config.max_generations - full_coop_idx - 1)

            # Ensure all arrays are exactly max_generations length
            coop_counts = coop_counts[:self.config.max_generations]
            costs = costs[:self.config.max_generations]
            welfare = welfare[:self.config.max_generations]

            generations = range(self.config.max_generations)

            # Convert to percentages
            c_percentage = [(count / population_size) * 100 for count in coop_counts]

            # Row 0: Frequency
            ax_freq = axes[0, col_idx]
            ax_freq.fill_between(generations, 0, c_percentage, color=C_COLOR, alpha=1.0,
                                linewidth=0.5, edgecolor=C_COLOR, label='C')
            ax_freq.fill_between(generations, c_percentage, 100, color=D_COLOR, alpha=1.0,
                                linewidth=0.5, edgecolor=D_COLOR, label='D')
            ax_freq.set_ylim(0, 100)
            ax_freq.set_xlim(0, self.config.max_generations)
            ax_freq.set_yticks([0, 20, 40, 60, 80, 100])
            ax_freq.set_xticks([0, 10, 20, 30, 40, self.config.max_generations])
            ax_freq.tick_params(labelsize=8)
            ax_freq.grid(True, alpha=0.35, linewidth=0.5, color='gray')

            if col_idx == 0:
                ax_freq.set_ylabel('frequency', fontsize=10)
            if col_idx == n_cols - 1:
                ax_freq.legend(loc='upper right', fontsize=8, frameon=True, fancybox=False)

            ax_freq.set_title(f'pc={pc_value}', fontsize=11, fontweight='bold')

            # Row 1: Cost
            ax_cost = axes[1, col_idx]
            ax_cost.fill_between(generations, costs, color=COST_COLOR, alpha=1.0,
                                linewidth=0.5, edgecolor=COST_COLOR)
            ax_cost.set_xlim(0, self.config.max_generations)
            ax_cost.set_xticks([0, 10, 20, 30, 40, self.config.max_generations])
            ax_cost.tick_params(labelsize=8)
            ax_cost.grid(True, alpha=0.35, linewidth=0.5, color='gray')

            if col_idx == 0:
                ax_cost.set_ylabel('cost', fontsize=10)

            # Row 2: Social Welfare
            ax_welfare = axes[2, col_idx]
            ax_welfare.fill_between(generations, welfare, color=WELFARE_COLOR, alpha=0.8,
                                   linewidth=0.5)
            ax_welfare.set_xlim(0, self.config.max_generations)
            ax_welfare.set_xlabel('generation', fontsize=10)
            ax_welfare.set_xticks([0, 10, 20, 30, 40, self.config.max_generations])
            ax_welfare.tick_params(labelsize=8)
            ax_welfare.grid(True, alpha=0.35, linewidth=0.5, color='gray')

            if col_idx == 0:
                ax_welfare.set_ylabel('SW (a=1)', fontsize=10)

        # Save plot
        if output_filename is None:
            pc_str = '_'.join(pc_values)
            output_filename = f"{data_dir}/ijcai_fig1_grid_theta{theta_value}_seed{seed}.png"

        plt.savefig(output_filename, dpi=self.config.dpi, bbox_inches='tight')
        print(f"IJCAI Figure 1 grid plot saved to: {output_filename}")

        if show_plot:
            plt.show()
        else:
            plt.close()

        return output_filename

    def plot_neb_figure1_grid(
        self,
        data_dir: str,
        nc_values: List[int],
        theta_value: float,
        population_size: int = 10000,
        output_filename: Optional[str] = None,
        show_plot: bool = False
    ) -> str:
        n_cols = len(nc_values)

        # IJCAI paper colors
        C_COLOR = self.config.IJCAI_COLORS['cooperator']
        D_COLOR = self.config.IJCAI_COLORS['defector']
        COST_COLOR = self.config.IJCAI_COLORS['cost']
        WELFARE_COLOR = self.config.IJCAI_COLORS['welfare']

        # Create figure: 3 rows × n_cols columns
        fig, axes = plt.subplots(3, n_cols, figsize=(4*n_cols, 6),
                                gridspec_kw={'hspace': 0.3, 'wspace': 0.25})

        # Ensure axes is 2D even with single column
        if n_cols == 1:
            axes = axes.reshape(-1, 1)

        for col_idx, nc_value in enumerate(nc_values):
            try:
                # Load NEB data
                neb_data = self.loader.load_neb_data(data_dir, nc_value)

                # Find the row for this theta value
                row = next((r for r in neb_data if abs(float(r['Theta']) - theta_value) < 0.01), None)

                if not row:
                    print(f"Warning: No data for nc={nc_value}, theta={theta_value}")
                    continue

                # Parse data using NEB-specific parser
                coop_counts = self.loader.parse_neb_metric(row, 'cooperator_frequency')
                costs = self.loader.parse_neb_metric(row, 'cost')
                welfare = self.loader.parse_neb_metric(row, 'social_welfare')

                # Find convergence point (where welfare becomes 0 after being non-zero)
                convergence_idx = None
                for i in range(len(welfare)):
                    if i > 0 and welfare[i] == 0 and welfare[i-1] > 0:
                        convergence_idx = i
                        break

                # If converged, pad with last non-zero value
                if convergence_idx is not None:
                    last_valid_welfare = welfare[convergence_idx - 1]
                    welfare = welfare[:convergence_idx] + [last_valid_welfare] * (self.config.max_generations - convergence_idx)

                    # Also ensure coop and cost are extended to max_generations
                    if len(coop_counts) < self.config.max_generations:
                        coop_counts = coop_counts + [coop_counts[-1]] * (self.config.max_generations - len(coop_counts))
                    if len(costs) < self.config.max_generations:
                        costs = costs + [costs[-1]] * (self.config.max_generations - len(costs))
                else:
                    # Normal padding if needed
                    actual_len = min(len(coop_counts), len(costs), len(welfare))
                    if actual_len < self.config.max_generations:
                        coop_counts = coop_counts[:actual_len] + [coop_counts[actual_len-1]] * (self.config.max_generations - actual_len)
                        costs = costs[:actual_len] + [costs[actual_len-1]] * (self.config.max_generations - actual_len)
                        welfare = welfare[:actual_len] + [welfare[actual_len-1]] * (self.config.max_generations - actual_len)

                # Ensure all arrays are exactly max_generations length
                coop_counts = coop_counts[:self.config.max_generations]
                costs = costs[:self.config.max_generations]
                welfare = welfare[:self.config.max_generations]

                generations = range(self.config.max_generations)

                # Convert to percentages
                c_percentage = [(count / population_size) * 100 for count in coop_counts]

                # Row 0: Frequency
                ax_freq = axes[0, col_idx]
                ax_freq.fill_between(generations, 0, c_percentage, color=C_COLOR, alpha=1.0,
                                    linewidth=0.5, edgecolor=C_COLOR, label='C')
                ax_freq.fill_between(generations, c_percentage, 100, color=D_COLOR, alpha=1.0,
                                    linewidth=0.5, edgecolor=D_COLOR, label='D')
                ax_freq.set_ylim(0, 100)
                ax_freq.set_xlim(0, self.config.max_generations)
                ax_freq.set_yticks([0, 20, 40, 60, 80, 100])
                ax_freq.set_xticks([0, 10, 20, 30, 40, self.config.max_generations])
                ax_freq.tick_params(labelsize=8)
                ax_freq.grid(True, alpha=0.35, linewidth=0.5, color='gray')

                if col_idx == 0:
                    ax_freq.set_ylabel('frequency', fontsize=10)
                if col_idx == n_cols - 1:
                    ax_freq.legend(loc='upper right', fontsize=8, frameon=True, fancybox=False)

                ax_freq.set_title(f'nc={nc_value}', fontsize=11, fontweight='bold')

                # Row 1: Cost
                ax_cost = axes[1, col_idx]
                ax_cost.fill_between(generations, costs, color=COST_COLOR, alpha=1.0,
                                    linewidth=0.5, edgecolor=COST_COLOR)
                ax_cost.set_xlim(0, self.config.max_generations)
                ax_cost.set_xticks([0, 10, 20, 30, 40, self.config.max_generations])
                ax_cost.tick_params(labelsize=8)
                ax_cost.grid(True, alpha=0.35, linewidth=0.5, color='gray')

                if col_idx == 0:
                    ax_cost.set_ylabel('cost', fontsize=10)

                # Row 2: Social Welfare
                ax_welfare = axes[2, col_idx]
                ax_welfare.fill_between(generations, welfare, color=WELFARE_COLOR, alpha=0.8,
                                       linewidth=0.5)
                ax_welfare.set_xlim(0, self.config.max_generations)
                ax_welfare.set_xlabel('generation', fontsize=10)
                ax_welfare.set_xticks([0, 10, 20, 30, 40, self.config.max_generations])
                ax_welfare.tick_params(labelsize=8)
                ax_welfare.grid(True, alpha=0.35, linewidth=0.5, color='gray')

                if col_idx == 0:
                    ax_welfare.set_ylabel('SW (a=1)', fontsize=10)

            except FileNotFoundError:
                print(f"Warning: Data file not found for nc={nc_value}")
                continue

        # Save plot
        if output_filename is None:
            nc_str = '_'.join(map(str, nc_values))
            output_filename = f"{data_dir}/neb_fig1_grid_theta{theta_value}.png"

        plt.savefig(output_filename, dpi=self.config.dpi, bbox_inches='tight')
        print(f"NEB Figure 1 grid plot saved to: {output_filename}")

        if show_plot:
            plt.show()
        else:
            plt.close()

        return output_filename

    def plot_ijcai_figure2_multiseed(
        self,
        data_dir: str,
        pc_values: List[str],
        theta_values: List[float],
        seeds: List[int],
        output_prefix: str = 'ijcai_fig2_multiseed',
        show_plot: bool = False
    ) -> Tuple[str, str]:
        # Define metrics with aggregation strategy
        metrics_config = {
            'cooperator_frequency': {'aggregation': 'final', 'label': 'Cooperation Frequency (%)'},
            'cost': {'aggregation': 'sum', 'label': 'Total Cost'},
            'social_welfare': {'aggregation': 'final', 'label': 'Social Welfare'}
        }

        # Initialize storage: [metric][seed][theta][pc]
        all_data = {metric: np.zeros((len(seeds), len(theta_values), len(pc_values)))
                   for metric in metrics_config.keys()}

        print(f"Loading data for {len(seeds)} seeds × {len(theta_values)} theta × {len(pc_values)} pc...")

        # Load all data
        for seed_idx, seed in enumerate(seeds):
            print(f"  Processing seed {seed}...")
            for pc_idx, pc in enumerate(pc_values):
                for metric, config in metrics_config.items():
                    try:
                        metric_data = self.loader.load_metric_data(data_dir, metric, pc, seed)

                        for theta_idx, theta in enumerate(theta_values):
                            row = next((r for r in metric_data
                                      if abs(float(r['Theta']) - theta) < 0.01), None)

                            if row:
                                values = self.loader.parse_metric_from_row(row, metric)

                                # Aggregate based on config
                                if config['aggregation'] == 'sum':
                                    value = sum(values) if values else 0
                                else:  # 'final'
                                    value = values[-1] if values else 0

                                # Convert cooperator frequency to percentage
                                if metric == 'cooperator_frequency':
                                    value = (value / self.config.default_population_size) * 100

                                all_data[metric][seed_idx, theta_idx, pc_idx] = value

                    except FileNotFoundError:
                        print(f"    Warning: Data file not found for seed={seed}, pc={pc}, metric={metric}")

        # Compute statistics
        mean_data = {metric: np.mean(all_data[metric], axis=0) for metric in metrics_config.keys()}
        std_data = {metric: np.std(all_data[metric], axis=0) for metric in metrics_config.keys()}

        # Plot MEAN heatmap
        fig_mean, axes = plt.subplots(1, 3, figsize=(18, 5))

        for idx, (metric, config) in enumerate(metrics_config.items()):
            ax = axes[idx]
            data = mean_data[metric]

            self._plot_single_metric_heatmap(
                ax=ax,
                data=data,
                pc_values=pc_values,
                theta_values=theta_values,
                metric=metric,
                title=config['label'],
                colorbar_label='Mean Value',
                cmap='plasma',
                annotation_fontsize=7
            )

        plt.tight_layout()

        output_mean = f'{data_dir}/{output_prefix}_mean_seeds{seeds[0]}-{seeds[-1]}.png'
        plt.savefig(output_mean, dpi=self.config.dpi, bbox_inches='tight')
        print(f"Mean heatmap saved to: {output_mean}")

        if show_plot:
            plt.show()
        else:
            plt.close()

        # Plot STD heatmap
        fig_std, axes = plt.subplots(1, 3, figsize=(18, 5))

        for idx, (metric, config) in enumerate(metrics_config.items()):
            ax = axes[idx]
            data = std_data[metric]

            self._plot_single_metric_heatmap(
                ax=ax,
                data=data,
                pc_values=pc_values,
                theta_values=theta_values,
                metric=metric,
                title=f'{config["label"]} (Std Dev)',
                colorbar_label='Std Dev',
                cmap='Reds',
                annotation_fmt='.1f',
                annotation_fontsize=7
            )

        plt.tight_layout()

        output_std = f'{data_dir}/{output_prefix}_std_seeds{seeds[0]}-{seeds[-1]}.png'
        plt.savefig(output_std, dpi=self.config.dpi, bbox_inches='tight')
        print(f"Std Dev heatmap saved to: {output_std}")

        if show_plot:
            plt.show()
        else:
            plt.close()

        return output_mean, output_std

    def plot_efficiency_comparison(
        self,
        data_dir: str,
        pc_values: List[str],
        seeds: List[int],
        a_values: List[float] = [0.5, 0.75, 1.0, 1.5, 2.0],
        theta_range: Tuple[float, float] = (4.0, 5.0),
        output_filename: Optional[str] = None,
        show_plot: bool = False
    ) -> str:
        # Get theta values from first available data file
        all_theta_values = self.loader.get_unique_theta_values(data_dir, pc_values[0], seeds[0])
        # Filter to specified range
        theta_values = [t for t in all_theta_values if theta_range[0] <= t <= theta_range[1]]

        print(f"Loading data for {len(seeds)} seeds × {len(theta_values)} theta × {len(pc_values)} pc...")

        metrics = ['cooperator_frequency', 'cost', 'social_welfare', 'population_payoff']
        all_data = {metric: np.zeros((len(seeds), len(theta_values), len(pc_values)))
                   for metric in metrics}

        for seed_idx, seed in enumerate(seeds):
            print(f"  Processing seed {seed}...")
            for pc_idx, pc in enumerate(pc_values):
                for metric in metrics:
                    try:
                        metric_data = self.loader.load_metric_data(data_dir, metric, pc, seed)

                        for theta_idx, theta in enumerate(theta_values):
                            row = next((r for r in metric_data if abs(float(r['Theta']) - theta) < 0.01), None)

                            if row:
                                values = self.loader.parse_metric_from_row(row, metric)
                                # Use final value for all metrics
                                value = values[-1] if values else 0

                                # Convert cooperator frequency to percentage
                                if metric == 'cooperator_frequency':
                                    value = (value / self.config.default_population_size) * 100

                                all_data[metric][seed_idx, theta_idx, pc_idx] = value

                    except FileNotFoundError:
                        continue

        # Compute mean across seeds for original a=1 data
        base_coop = np.mean(all_data['cooperator_frequency'], axis=0)
        base_cost_a1 = np.mean(all_data['cost'], axis=0)
        base_payoff = np.mean(all_data['population_payoff'], axis=0)

        # Create figure: 3 rows × len(a_values) columns
        fig, axes = plt.subplots(3 - 1, len(a_values), figsize=(6 * len(a_values), 10))

        metric_labels = {
            'cooperator_frequency': 'Cooperation Frequency (%)',
            'cost': 'Total Cost',
            'social_welfare': 'Social Welfare'
        }

        # Plot each a value
        for a_idx, a in enumerate(a_values):
            print(f"\nGenerating plots for a={a}...")

            # Recalculate cost and social welfare for this 'a'
            cost_a = base_cost_a1 / a  # Cost scales inversely with efficiency
            welfare_a = base_payoff - cost_a

            # Row 0: Cooperation frequency (doesn't change with 'a')
            # if we plot only >=90%, we only care about the two later values
            # ax_coop = axes[0, a_idx]
            # self._plot_single_metric_heatmap(
            #     ax=ax_coop,
            #     data=base_coop,
            #     pc_values=pc_values,
            #     theta_values=theta_values,
            #     metric='cooperator_frequency',
            #     title=f'a = {a}',
            #     colorbar_label='Cooperation %',
            #     cmap='plasma',
            #     annotation_fontsize=7
            # )

            # Row 1: Cost (changes with 'a')
            # ax_cost = axes[1, a_idx]
            ax_cost = axes[0, a_idx]
            self._plot_single_metric_heatmap(
                ax=ax_cost,
                data=cost_a,
                pc_values=pc_values,
                theta_values=theta_values,
                metric='cost',
                title=f'a = {a}',
                # title='',  # No title for middle row
                colorbar_label='Total Cost',
                cmap='plasma',
                annotation_fontsize=7
            )

            # Row 2: Social Welfare (changes with 'a')
            # ax_welfare = axes[2, a_idx]
            ax_welfare = axes[1, a_idx]
            self._plot_single_metric_heatmap(
                ax=ax_welfare,
                data=welfare_a,
                pc_values=pc_values,
                theta_values=theta_values,
                metric='social_welfare',
                title='',  # No title for bottom row
                colorbar_label='Social Welfare',
                cmap='plasma',
                annotation_fontsize=7
            )

            # Add row labels on leftmost column
            if a_idx == 0:
                # ax_coop.set_ylabel(f'{metric_labels["cooperator_frequency"]}\n\nTheta (θ)',
                #                   fontsize=self.config.label_fontsize)
                ax_cost.set_ylabel(f'{metric_labels["cost"]}\n\nTheta (θ)',
                                  fontsize=self.config.label_fontsize)
                ax_welfare.set_ylabel(f'{metric_labels["social_welfare"]}\n\nTheta (θ)',
                                     fontsize=self.config.label_fontsize)

        plt.tight_layout()

        # Save plot
        if output_filename is None:
            output_filename = f"{data_dir}/efficiency_comparison_seeds{seeds[0]}-{seeds[-1]}.png"

        plt.savefig(output_filename, dpi=self.config.dpi, bbox_inches='tight')
        print(f"Efficiency comparison plot saved to: {output_filename}")

        if show_plot:
            plt.show()
        else:
            plt.close()

        return output_filename
