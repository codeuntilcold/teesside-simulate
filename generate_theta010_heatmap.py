from plotting_utils import SimulationPlotter
import argparse
import numpy as np


def generate_for_r(plotter, r, seeds, data_source, theta_values, show_plot):
    r_str = str(int(r)) if r == int(r) else str(r)

    # POP heatmap
    if data_source == 'go':
        pop_data_dir = f'data_go_theta010/pop_theta_pgg_r={r_str}'
        pop_pc_values = ['0.25', '0.5', '0.75', '0.9', '0.92', '0.94', '0.96', '0.98', '1']
    else:
        pop_data_dir = f'data_py_theta010/pop_theta_pgg_r={r}'
        pop_pc_values = ['0.25', '0.5', '0.75', '0.9', '0.92', '0.94', '0.96', '0.98', '1.0']

    print("=" * 60)
    print(f"Generating theta 0-10 heatmaps for POP strategy")
    print("=" * 60)
    print(f"  r={r}, seeds={seeds[0]}-{seeds[-1]}")
    print(f"  Data dir: {pop_data_dir}")
    print(f"  p_C values: {', '.join(pop_pc_values)}")

    try:
        output_prefix = f'pgg_r={r_str}_pop_theta010_{data_source}'
        mean_file, std_file = plotter.plot_ijcai_figure2_multiseed(
            data_dir=pop_data_dir,
            pc_values=pop_pc_values,
            theta_values=theta_values,
            seeds=seeds,
            param_name='pc',
            theta_range='0.0-10.0',
            output_dir='fig',
            output_prefix=output_prefix,
            show_plot=show_plot
        )
        print(f"\nPOP Mean heatmap: {mean_file}")
        print(f"POP Std heatmap: {std_file}")
    except Exception as e:
        print(f"  Error: {e}")

    # NEB heatmap
    if data_source == 'go':
        neb_data_dir = f'data_go_theta010/neb_theta_pgg_r={r_str}'
        neb_nc_values = ['1', '2', '3', '4']
    else:
        neb_data_dir = f'data_py_theta010/neb_theta_pgg_r={r}'
        neb_nc_values = ['1', '2', '3', '4']

    print("\n" + "=" * 60)
    print(f"Generating theta 0-10 heatmaps for NEB strategy")
    print("=" * 60)
    print(f"  r={r}, seeds={seeds[0]}-{seeds[-1]}")
    print(f"  Data dir: {neb_data_dir}")
    print(f"  n_C values: {', '.join(neb_nc_values)}")

    try:
        output_prefix = f'pgg_r={r_str}_neb_theta010_{data_source}'
        mean_file, std_file = plotter.plot_ijcai_figure2_multiseed(
            data_dir=neb_data_dir,
            pc_values=neb_nc_values,
            theta_values=theta_values,
            seeds=seeds,
            param_name='nc',
            theta_range='0.0-10.0',
            output_dir='fig',
            output_prefix=output_prefix,
            show_plot=show_plot
        )
        print(f"\nNEB Mean heatmap: {mean_file}")
        print(f"NEB Std heatmap: {std_file}")
    except Exception as e:
        print(f"  Error: {e}")


def main():
    parser = argparse.ArgumentParser(description='Generate theta 0-10 heatmaps (averaged across seeds).')
    parser.add_argument('--seed-start', type=int, default=0, help='Start seed')
    parser.add_argument('--seed-end', type=int, default=50, help='End seed (exclusive)')
    parser.add_argument('--data-source', choices=['go', 'py'], default='go', help='Data source: go or py')
    parser.add_argument('--show', action='store_true', help='Show plot interactively')
    args = parser.parse_args()

    plotter = SimulationPlotter()
    seeds = list(range(args.seed_start, args.seed_end))
    theta_values = list(np.arange(0, 10.1, 0.1))
    r_values = [1.5, 3, 4.5]

    for r in r_values:
        generate_for_r(plotter, r, seeds, args.data_source, theta_values, args.show)


if __name__ == "__main__":
    main()
