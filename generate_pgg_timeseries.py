from plotting_utils import SimulationPlotter
import argparse


def main():
    parser = argparse.ArgumentParser(description='Generate PGG time series figures (averaged across seeds).')
    parser.add_argument('--r', type=float, default=3.0, help='PGG multiplication factor')
    parser.add_argument('--theta', type=float, default=4.5, help='Theta value to plot')
    parser.add_argument('--seed-start', type=int, default=0, help='Start seed')
    parser.add_argument('--seed-end', type=int, default=50, help='End seed (exclusive)')
    parser.add_argument('--no-std', action='store_true', help='Hide std deviation shading')
    parser.add_argument('--show', action='store_true', help='Show plot interactively')
    args = parser.parse_args()

    plotter = SimulationPlotter()

    r = args.r
    theta = args.theta
    seeds = list(range(args.seed_start, args.seed_end))
    show_std = not args.no_std

    # Generate POP figures
    pop_data_dir = f'data/pop_theta_pgg_r={r}'
    pc_values = ['0.25', '0.5', '0.75', '1.0']

    print(f"Generating PGG time series for POP strategy (averaged over {len(seeds)} seeds)")
    print(f"  r={r}, theta={theta}")
    print(f"  p_C values: {', '.join(pc_values)}")
    print(f"  Data dir: {pop_data_dir}")

    pop_filename = plotter.plot_ijcai_figure1_grid_multiseed(
        data_dir=pop_data_dir,
        pc_values=pc_values,
        theta_value=theta,
        seeds=seeds,
        show_std=show_std,
        output_filename=f'fig/pgg_pop_timeseries_r={r}_theta={theta}_avg.png',
        show_plot=args.show
    )
    print(f"POP output: {pop_filename}\n")

    # Generate NEB figures
    neb_data_dir = f'data/neb_theta_pgg_r={r}'
    nc_values = ['1', '2', '3', '4']

    print(f"Generating PGG time series for NEB strategy (averaged over {len(seeds)} seeds)")
    print(f"  r={r}, theta={theta}")
    print(f"  n_C values: {', '.join(nc_values)}")
    print(f"  Data dir: {neb_data_dir}")

    neb_filename = plotter.plot_ijcai_figure1_grid_multiseed(
        data_dir=neb_data_dir,
        pc_values=nc_values,
        theta_value=theta,
        seeds=seeds,
        param_name='nc',
        show_std=show_std,
        output_filename=f'fig/pgg_neb_timeseries_r={r}_theta={theta}_avg.png',
        show_plot=args.show
    )
    print(f"NEB output: {neb_filename}")


if __name__ == "__main__":
    main()
