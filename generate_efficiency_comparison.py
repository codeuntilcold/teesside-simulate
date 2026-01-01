from plotting_utils import SimulationPlotter


def main():
    plotter = SimulationPlotter()
    data_dir = 'data/pop_theta'

    pc_values = [
            # '0.25', '0.5', '0.75',
            '0.9', '0.92', '0.94', '0.96', '0.98', '1.0',
            ]
    seeds = list(range(10))
    a_values = [0.5, 1.0, 1.5]

    output_file = plotter.plot_efficiency_comparison(
        data_dir=data_dir,
        pc_values=pc_values,
        seeds=seeds,
        a_values=a_values,
        theta_filter=(4.0, 5.0),
        output_dir='fig'
    )

    print(f"{output_file=}")


if __name__ == "__main__":
    main()
