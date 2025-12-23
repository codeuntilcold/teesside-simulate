from plotting_utils import SimulationPlotter


if __name__ == "__main__":
    plotter = SimulationPlotter()
    data_dir = 'data/pop_theta'

    pc_values = ['0.25', '0.5', '0.75', '1.0']
    theta = 4.5
    seed = 0

    print(f"p_C values: {', '.join(pc_values)}")
    print(f"{theta=}")
    print(f"{seed=}")

    filename = plotter.plot_ijcai_figure1_grid(
        data_dir=data_dir,
        pc_values=pc_values,
        theta_value=theta,
        seed=seed,
        show_plot=False
    )

    print(f"{filename=}")
