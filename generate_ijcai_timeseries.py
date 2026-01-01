from plotting_utils import SimulationPlotter


if __name__ == "__main__":
    plotter = SimulationPlotter()
    data_dir = 'data/pop_theta'

    pc_values = ['0.25', '0.5', '0.75', '1.0']
    nc_values = [1, 2, 3, 4]
    theta = 4.5

    print(f"p_C values: {', '.join(pc_values)}")
    print(f"{theta=}")

    filename = plotter.plot_ijcai_figure1_grid(
        data_dir='data/pop_theta',
        pc_values=pc_values,
        theta_value=theta,
        # TODO: do average, don't do seed
        seed=0,
        show_plot=False
    )

    print(f"POP: {pc_values=} {theta=} {filename=}")

    filename = plotter.plot_neb_figure1_grid(
        data_dir='data/neb_theta',
        nc_values=nc_values,
        theta_value=theta,
        show_plot=False
    )

    print(f"NEB: {nc_values=} {theta=} {filename=}")
