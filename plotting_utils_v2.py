import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Dict, Tuple, Optional, Any, TypedDict
from dataclasses import dataclass


class MetricMatrices(TypedDict):
    """Per-strategy-param metric matrices, shape [theta, sp]. Cost and welfare are
    cumulative (summed over generations); coop_freq is final-gen %."""
    cost: np.ndarray
    welfare: np.ndarray
    coop_freq: np.ndarray


@dataclass
class CostWelfareTradeoff:
    """Diff-plot landmarks and normalised series for one strategy-param column."""
    sw_a: np.ndarray
    cost_norm: np.ndarray
    sw_norm: np.ndarray
    cost_min_theta: float
    sw_max_theta: float
    coef: int  # -1, 0, or 1: order of cost_min_theta vs sw_max_theta
    span: float  # signed Δθ between the two landmarks


@dataclass
class OptimalRecord:
    """One (a, game, strategy, game_param, sp) cell for the optimal-θ summary."""
    a: float
    game: str          # 'pd' or 'pgg'
    strategy: str      # 'pop' or 'neb'
    game_param: str    # e.g. 'b=1.2', 'r=3.0'
    sp: str            # e.g. 'pc=0.92', 'nc=4'
    theta_sw: float
    theta_cost: float
    feasible: bool
    max_coop: float


COLORS = {
    'cooperator': '#6B7FD7',
    'defector': '#FF7875',
    'cost': '#B88BBD',
    'welfare': '#F18F01',
}
MAX_GENERATIONS = 50
POPULATION_SIZE = 10000
DPI = 300
LABEL_FONTSIZE = 16
TICK_FONTSIZE = 13
TITLE_FONTSIZE = 15
COLOR_MAP = "inferno"
COLOR_INTERP = "bilinear"


def _apply_rcparams():
    """Set matplotlib title/label defaults so per-call fontsize/fontweight kwargs aren't needed.

    Note: xtick/ytick labelsize stays at matplotlib default to avoid disturbing colorbar
    and other axes that didn't originally override it. The time-series helper restores 11pt
    explicitly via tick_params; heatmap data axes do likewise via set_xticklabels(fontsize=).
    """
    import matplotlib as mpl
    mpl.rcParams.update({
        'axes.titlesize': TITLE_FONTSIZE,
        'axes.titleweight': 'bold',
        'axes.labelsize': LABEL_FONTSIZE,
    })


_apply_rcparams()


# ---------------------------------------------------------------------------
# Pure data transforms (no matplotlib calls, no I/O)
# ---------------------------------------------------------------------------


def pad_series(arr, target_len: int, pad_value: Optional[float] = None) -> np.ndarray:
    """Pad with last value (or pad_value), or truncate, to target_len."""
    arr = np.asarray(arr, dtype=float)
    if len(arr) >= target_len:
        return arr[:target_len]
    fill = pad_value if pad_value is not None else arr[-1]
    return np.concatenate([arr, np.full(target_len - len(arr), fill)])


def prepare_timeseries_arrays(
    sp_data: Dict[str, Dict[float, Dict[str, List[float]]]],
    theta: float,
    max_gen: int,
    pop_size: int,
) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
    """Pad/truncate timeseries to max_gen, convert coop to %, clip cost std band ≥ 0.

    Returns None if cooperator_frequency for the given theta is missing.
    """
    coop = sp_data.get('cooperator_frequency', {}).get(theta)
    cost = sp_data.get('cost', {}).get(theta)
    welfare = sp_data.get('social_welfare', {}).get(theta)
    if not coop:
        return None

    coop_mean = pad_series(coop['mean'], max_gen)
    coop_std = pad_series(coop['std'], max_gen)
    cost_mean = pad_series(cost['mean'], max_gen, pad_value=0)
    cost_std = pad_series(cost['std'], max_gen, pad_value=0)
    welfare_mean = pad_series(welfare['mean'], max_gen)
    welfare_std = pad_series(welfare['std'], max_gen)

    return {
        'coop': {
            'mean': (coop_mean / pop_size) * 100,
            'std': (coop_std / pop_size) * 100,
        },
        'cost': {
            'mean': cost_mean,
            'lower': np.maximum(cost_mean - cost_std, 0),
            'upper': cost_mean + cost_std,
        },
        'welfare': {
            'mean': welfare_mean,
            'lower': welfare_mean - welfare_std,
            'upper': welfare_mean + welfare_std,
        },
    }


def adjusted_welfare_for_efficiency(welfare: np.ndarray, cost: np.ndarray, a: float) -> np.ndarray:
    """Heatmap formula: welfare + (a - 1) * cost."""
    return welfare + (a - 1) * cost


def analyze_cost_welfare_tradeoff(
    cost: np.ndarray,
    welfare: np.ndarray,
    coop_freq: np.ndarray,
    theta_values: List[float],
    a: float,
    coop_threshold: float = 90.0,
) -> CostWelfareTradeoff:
    """Diff-plot transform for one strategy column. Inputs are 1D over theta.

    Per the paper: institutional cost is θ·|C_inv|, independent of a. Recorded
    welfare at sim-time a=1 is F_base; SW at hypothetical a' is welfare+(a-1)*cost.
    """
    sw_a = welfare + (a - 1) * cost

    high_coop_idx = np.where(coop_freq > coop_threshold)[0]
    cost_min_idx = high_coop_idx[np.argmin(cost[high_coop_idx])]
    sw_max_idx = int(np.argmax(sw_a))

    cost_min_theta = theta_values[cost_min_idx]
    sw_max_theta = theta_values[sw_max_idx]

    if cost_min_theta == sw_max_theta:
        coef = 0
    elif cost_min_theta > sw_max_theta:
        coef = -1
    else:
        coef = 1

    return CostWelfareTradeoff(
        sw_a=sw_a,
        cost_norm=(cost - cost.min()) / (cost.max() - cost.min()),
        sw_norm=(sw_a - sw_a.min()) / (sw_a.max() - sw_a.min()),
        cost_min_theta=cost_min_theta,
        sw_max_theta=sw_max_theta,
        coef=coef,
        span=coef * (sw_max_theta - cost_min_theta),
    )


_GAME_LABELS = {'pd': 'PD', 'pgg': 'PGG'}


def latex_sp(sp: str) -> str:
    """`pc=1` -> `$p_C=1$`; `nc=4` -> `$n_C=4$`; passthrough otherwise."""
    if sp.startswith('pc='):
        return '$p_C=' + sp[3:] + '$'
    if sp.startswith('nc='):
        return '$n_C=' + sp[3:] + '$'
    return sp


def latex_game_param(gp: str) -> str:
    """`b=1.8` -> `$b=1.8$`, `r=3.0` -> `$r=3.0$`."""
    return '$' + gp + '$'


_OPTIMAL_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c']
_OPTIMAL_MARKERS = ['o', 's', '^']
_OPTIMAL_SW_OFFSETS = [(-18, 8), (5, 8), (12, -10)]
_OPTIMAL_COST_OFFSETS = [(-18, -12), (5, -12), (12, 4)]


# ---------------------------------------------------------------------------
# Pure viz helpers
# ---------------------------------------------------------------------------


def style_timeseries_ax(ax, max_gen: int, tick_fontsize: int = 11):
    """Common styling for the time-series x-axis and grid."""
    ax.set_xlim(0, max_gen)
    ax.set_xticks([0, 10, 20, 30, 40, max_gen])
    ax.tick_params(labelsize=tick_fontsize)
    ax.grid(True, alpha=0.35, linewidth=0.5, color='gray')


def fill_band(ax, x, y, color: str, alpha: float = 1.0):
    """Fill the area between 0 and y with a single colour."""
    ax.fill_between(x, y, color=color, alpha=alpha, linewidth=0.5, edgecolor=color)


def fill_stacked_to_100(ax, x, y, lower_color: str, upper_color: str,
                        labels: Optional[Tuple[str, str]] = None):
    """Fill 0→y in lower_color and y→100 in upper_color (stacked-frequency band)."""
    lo_label, hi_label = labels or (None, None)
    ax.fill_between(x, 0, y, color=lower_color, alpha=1.0, linewidth=0.5,
                    edgecolor=lower_color, label=lo_label)
    ax.fill_between(x, y, 100, color=upper_color, alpha=1.0, linewidth=0.5,
                    edgecolor=upper_color, label=hi_label)


def _draw_diff_panel(ax, tradeoff: CostWelfareTradeoff, coop: np.ndarray,
                     sorted_thetas: List[float], sp_label: str, a: float,
                     *, set_ylim_top: bool):
    """Draw one (sp, a) cell of the diff plot."""
    cost_min_theta = tradeoff.cost_min_theta
    sw_max_theta = tradeoff.sw_max_theta

    ax.plot(sorted_thetas, tradeoff.cost_norm, color='lightcoral', label='Total Cost')
    ax.plot(sorted_thetas, tradeoff.sw_norm, color='deepskyblue', label='Social Welfare')
    ax.plot(sorted_thetas, coop / 100, linestyle='-.', color='darkcyan', label='Coop Freq')

    span = tradeoff.span
    ax.axhline(0.9, xmin=0.2 / (span + 0.4), xmax=(span + 0.2) / (span + 0.4),
               color='limegreen', linestyle='--')
    ax.axvline(cost_min_theta, linestyle=(0, (3, 3)), color='lightcoral')
    ax.axvline(sw_max_theta, linestyle=(2, (3, 3)), color='deepskyblue')
    ax.text(cost_min_theta + 0.01, 0.5, round(cost_min_theta, 1), rotation=90, va='center')
    ax.text(sw_max_theta + 0.01, 0.5, round(sw_max_theta, 1), rotation=90, va='center')
    ax.text((cost_min_theta + sw_max_theta) / 2, 0.92, 'Δθ', va='center')
    ax.plot([cost_min_theta], [0.9], color='limegreen', marker='o')
    ax.plot([sw_max_theta], [0.9], color='limegreen', marker='o')

    if tradeoff.coef == 1:
        ax.set_xlim(cost_min_theta - 0.2, sw_max_theta + 0.2)
    elif tradeoff.coef == -1:
        ax.set_xlim(sw_max_theta - 0.2, cost_min_theta + 0.2)
    if set_ylim_top:
        ax.set_ylim(0, 1.1)

    ax.set_xlabel('Per-individual investment cost, θ')
    ax.set_ylabel('Normalized Value')
    ax.set_title(f'{latex_sp(sp_label)}, $a={a}$')


def _draw_optimal_panel(ax, cell_records: List[OptimalRecord],
                        *, strategy: str, a: float,
                        is_first_row: bool, is_first_col: bool):
    """Draw one (a, strategy) cell of the optimal-θ summary."""
    if not cell_records:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return

    by_game_param: Dict[str, List[OptimalRecord]] = {}
    for r in cell_records:
        by_game_param.setdefault(r.game_param, []).append(r)

    x_vals: List[float] = []
    for idx, game_param in enumerate(sorted(by_game_param.keys())):
        sp_records = sorted(by_game_param[game_param], key=lambda r: float(r.sp.split('=')[1]))
        x_vals = [float(r.sp.split('=')[1]) for r in sp_records]
        color = _OPTIMAL_COLORS[idx % len(_OPTIMAL_COLORS)]
        marker = _OPTIMAL_MARKERS[idx % len(_OPTIMAL_MARKERS)]
        sw_off = _OPTIMAL_SW_OFFSETS[idx % len(_OPTIMAL_SW_OFFSETS)]
        cost_off = _OPTIMAL_COST_OFFSETS[idx % len(_OPTIMAL_COST_OFFSETS)]

        for x, r in zip(x_vals, sp_records):
            if not r.feasible:
                continue

            ax.scatter([x], [r.theta_sw], marker=marker, color=color, s=60, zorder=3)
            ax.annotate(f'{r.theta_sw:.1f}', (x, r.theta_sw),
                        textcoords='offset points', xytext=sw_off,
                        fontsize=7, color=color)

            ax.scatter([x], [r.theta_cost], marker=marker, facecolors='none',
                       edgecolors=color, s=60, linewidths=1.5, zorder=3)
            ax.annotate(f'{r.theta_cost:.1f}', (x, r.theta_cost),
                        textcoords='offset points', xytext=cost_off,
                        fontsize=7, color=color)

    ax.set_xlabel('$p_C$' if strategy == 'pop' else '$n_C$')
    ax.set_xticks(x_vals)
    ax.grid(True, alpha=0.3)
    ax.autoscale(axis='y')
    y_min, y_max = ax.get_ylim()
    margin = (y_max - y_min) * 0.15
    ax.set_ylim(y_min - margin, y_max + margin)

    if is_first_col:
        ax.set_ylabel(f'a={a}\nθ*')
    if is_first_row:
        strategy_labels = {'pop': 'POP', 'neb': 'NEB'}
        ax.set_title(strategy_labels[strategy], fontweight='normal')


def _make_optimal_legend(game_param_labels: List[str]):
    """Build the per-game-param + filled-vs-hollow legend."""
    from matplotlib.lines import Line2D
    handles = []
    for idx, label in enumerate(game_param_labels):
        handles.append(Line2D([0], [0], marker=_OPTIMAL_MARKERS[idx], color='w',
                              markerfacecolor=_OPTIMAL_COLORS[idx], markersize=8,
                              label=latex_game_param(label)))
    handles.append(Line2D([0], [0], marker='o', color='w',
                          markerfacecolor='gray', markersize=8, label='Filled = SW'))
    handles.append(Line2D([0], [0], marker='o', color='w',
                          markerfacecolor='none', markeredgecolor='gray',
                          markeredgewidth=1.5, markersize=8, label='Hollow = Cost'))
    return handles


def finalize_figure(fig, output_filename: Optional[str], dpi: int, show_plot: bool):
    """Save figure to file (and close) or show interactively."""
    if output_filename:
        save_figure(fig, output_filename, dpi, show_plot)
    else:
        plt.show()


def save_figure(fig, output_path: str, dpi: int = 300, show: bool = False, message: str = None):
    """Save figure, print message, optionally show, then close."""
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    import subprocess
    subprocess.run(['mogrify', '-resize', '1920x1920>', '-quality', '90', output_path], check=True)
    if message:
        print(message)
    else:
        print(f"Saved: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_timeseries_from_agg(
    agg_data: Dict[str, Dict[str, Dict[float, Dict[str, List[float]]]]],
    strategy_params: List[str],
    theta: float,
    title: str = '',
    show_std: bool = True,
    output_filename: Optional[str] = None,
    show_plot: bool = False
) -> str:
    """
    Plot timeseries grid from pre-aggregated data.

    Args:
        agg_data: {strategy_param: {metric: {theta: {'mean': [...], 'std': [...]}}}}
        strategy_params: List of strategy param labels (e.g. ['pc=0.25', 'pc=0.5'])
        theta: Theta value to plot
        title: Overall figure title
        show_std: Whether to show standard deviation bands
    """
    n_cols = len(strategy_params)
    max_gen = MAX_GENERATIONS

    fig, axes = plt.subplots(3, n_cols, figsize=(4 * n_cols, 6),
                             gridspec_kw={'hspace': 0.3, 'wspace': 0.25})
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    colors = COLORS
    generations = np.arange(max_gen)
    tick_fontsize = TICK_FONTSIZE

    for col_idx, sp in enumerate(strategy_params):
        arrays = prepare_timeseries_arrays(
            agg_data.get(sp, {}), theta, max_gen, POPULATION_SIZE
        )
        if arrays is None:
            print(f"Warning: theta={theta} not found for {sp}")
            continue

        coop, cost, welfare = arrays['coop'], arrays['cost'], arrays['welfare']
        is_first = col_idx == 0

        ax_freq = axes[0, col_idx]
        fill_stacked_to_100(ax_freq, generations, coop['mean'],
                            colors['cooperator'], colors['defector'], labels=('C', 'D'))
        style_timeseries_ax(ax_freq, max_gen, tick_fontsize)
        ax_freq.set_ylim(0, 100)
        ax_freq.set_yticks([0, 20, 40, 60, 80, 100])
        if is_first:
            ax_freq.set_ylabel('frequency', fontsize=12)
        if show_std:
            ax_freq.fill_between(generations, coop['mean'] - coop['std'],
                                 coop['mean'] + coop['std'], color='white', alpha=0.3)
        if col_idx == n_cols - 1:
            ax_freq.legend(loc='upper right', fontsize=10, frameon=True, fancybox=False)
        ax_freq.set_title(sp)

        ax_cost = axes[1, col_idx]
        fill_band(ax_cost, generations, cost['mean'], colors['cost'])
        style_timeseries_ax(ax_cost, max_gen, tick_fontsize)
        if is_first:
            ax_cost.set_ylabel('cost', fontsize=12)
        if show_std:
            ax_cost.fill_between(generations, cost['lower'], cost['upper'],
                                 color=colors['cost'], alpha=0.3)

        ax_welfare = axes[2, col_idx]
        fill_band(ax_welfare, generations, welfare['mean'], colors['welfare'])
        style_timeseries_ax(ax_welfare, max_gen, tick_fontsize)
        ax_welfare.set_xlabel('generation', fontsize=12)
        if is_first:
            ax_welfare.set_ylabel('SW (a=1)', fontsize=12)
        if show_std:
            ax_welfare.fill_between(generations, welfare['lower'], welfare['upper'],
                                    color=colors['welfare'], alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=16, y=1.02)

    finalize_figure(fig, output_filename, DPI, show_plot)
    return output_filename

def plot_heatmap_grid_from_agg(
    data_matrices: MetricMatrices,
    strategy_params: List[str],
    theta_values: List[float],
    a_values: List[float],
    strategy: str,
    title: str = '',
    output_filename: Optional[str] = None,
    show_plot: bool = False
) -> str:
    """
    Plot heatmap grid for different 'a' values (cost efficiency).

    Args:
        data_matrices: {'cost': array, 'welfare': array} with shape [theta, param]
        strategy: 'pop' or 'neb'
        a_values: List of efficiency values to plot
    """
    n_cols = len(a_values)
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 8))

    xlabel = '$p_C$' if strategy == 'pop' else '$n_C$'
    x_tick_labels = [sp.split('=')[1] for sp in strategy_params]
    tick_step = max(1, len(theta_values) // min(11, len(theta_values)))
    y_tick_pos = list(range(0, len(theta_values), tick_step))
    y_tick_labels = [f"{theta_values[i]:.1f}" for i in y_tick_pos]

    def render(ax, matrix, panel_title, show_ylabel=False, vmin=None, vmax=None):
        im = ax.imshow(matrix, aspect='auto', cmap=COLOR_MAP,
                       origin='lower', interpolation=COLOR_INTERP,
                       vmin=vmin, vmax=vmax)
        ax.set_title(panel_title)
        ax.set_xticks(range(len(strategy_params)))
        ax.set_xticklabels(x_tick_labels, fontsize=TICK_FONTSIZE)
        ax.set_yticks(y_tick_pos)
        ax.set_yticklabels(y_tick_labels, fontsize=TICK_FONTSIZE)
        ax.set_xlabel(xlabel)
        if show_ylabel:
            ax.set_ylabel('θ')
        plt.colorbar(im, ax=ax)

    # Top row: cooperation level, total cost, (empty slots for a_values beyond index 1)
    render(axes[0, 0], data_matrices['coop_freq'], 'Cooperation level (%)', show_ylabel=True)
    render(axes[0, 1], data_matrices['cost'], 'Total Cost')
    for col_idx in range(2, n_cols):
        axes[0, col_idx].axis('off')

    # Bottom row: SW for each a, sharing colormap range so per-a magnitudes are comparable.
    sw_arrays = [adjusted_welfare_for_efficiency(data_matrices['welfare'], data_matrices['cost'], a)
                 for a in a_values]
    sw_vmin = min(arr.min() for arr in sw_arrays)
    sw_vmax = max(arr.max() for arr in sw_arrays)
    for col_idx, (a, welfare_a) in enumerate(zip(a_values, sw_arrays)):
        render(axes[1, col_idx], welfare_a,
               f'Social Welfare ($a = {a}$)',
               show_ylabel=(col_idx == 0),
               vmin=sw_vmin, vmax=sw_vmax)

    if title:
        fig.suptitle(title, fontsize=16, y=1.02)

    plt.tight_layout()

    finalize_figure(fig, output_filename, DPI, show_plot)
    return output_filename

def plot_diff_plot_from_agg(
    data_matrices: MetricMatrices,
    strategy_params: List[str],
    theta_values: List[float],
    a_values: List[float],
    strategy: str,
    title: str = '',
    output_filename: Optional[str] = None,
    show_plot: bool = False
):
    """
    Plot different line plot from pre-aggregated final values.

    POP variant (single row) plots only the last strategy param;
    NEB variant (two rows) plots the last two.

    Args:
        data_matrices: {'cost': array, 'welfare': array, 'coop_freq': array} with shape [theta, param]
        strategy: 'pop' or 'neb'
        strategy_params: List of strategy param labels
        theta_values: List of theta values
        title: Plot title
    """
    is_pop = strategy == 'pop'
    if is_pop:
        param_indices = [len(strategy_params) - 1]
        figsize = (15, 4)
    else:
        param_indices = [len(strategy_params) - 2, len(strategy_params) - 1]
        figsize = (15, 8)

    n_rows, n_cols = len(param_indices), len(a_values)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    sorted_thetas = sorted(theta_values)

    for row, sp_idx in enumerate(param_indices):
        for col, a in enumerate(a_values):
            tradeoff = analyze_cost_welfare_tradeoff(
                cost=data_matrices['cost'][:, sp_idx],
                welfare=data_matrices['welfare'][:, sp_idx],
                coop_freq=data_matrices['coop_freq'][:, sp_idx],
                theta_values=theta_values,
                a=a,
            )
            _draw_diff_panel(
                axes[row, col], tradeoff,
                coop=data_matrices['coop_freq'][:, sp_idx],
                sorted_thetas=sorted_thetas,
                sp_label=strategy_params[sp_idx],
                a=a,
                set_ylim_top=is_pop,
            )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1))

    if title:
        fig.suptitle(title, fontsize=16, y=1.05)

    plt.tight_layout()

    finalize_figure(fig, output_filename, DPI, show_plot)
    return output_filename

def plot_optimal_theta_summary(
    records: List[OptimalRecord],
    a_values: List[float],
    title: str = '',
    output_filename: Optional[str] = None,
    show_plot: bool = False
):
    """
    Plot grid showing θ* vs strategy params. Scatter only (no lines).
    Filled markers = SW optimal, hollow markers = Cost optimal.
    Rows = a values, Columns = strategy. One figure per game.
    """
    games = sorted({r.game for r in records})
    output_files = []

    for game in games:
        n_rows = len(a_values)
        fig, axes = plt.subplots(n_rows, 2, figsize=(10, 4 * n_rows))
        if n_rows == 1:
            axes = axes[np.newaxis, :]

        for row_idx, a in enumerate(a_values):
            for col, strategy in enumerate(['pop', 'neb']):
                cell = [r for r in records if r.game == game and r.a == a and r.strategy == strategy]
                _draw_optimal_panel(
                    axes[row_idx, col],
                    cell,
                    strategy=strategy, a=a,
                    is_first_row=(row_idx == 0),
                    is_first_col=(col == 0),
                )

        game_params = sorted({r.game_param for r in records if r.game == game},
                             key=lambda gp: float(gp.split('=')[1]))
        fig.legend(handles=_make_optimal_legend(game_params),
                   loc='upper center', ncol=len(game_params) + 2, bbox_to_anchor=(0.5, 1.0),
                   fontsize=12, frameon=True)

        game_title = f'{title} - {_GAME_LABELS[game]}' if title else _GAME_LABELS[game]
        fig.suptitle(game_title, fontsize=16, y=1.05)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if output_filename:
            base, ext = os.path.splitext(output_filename)
            out = f'{base}_{game}{ext}'
            output_files.append(out)
        else:
            out = None
        finalize_figure(fig, out, DPI, show_plot)

    return output_files
