import numpy as np
from plotting_utils import SimulationPlotter


def main():
    plotter = SimulationPlotter()
    data_dir = 'data/pop_theta'
    seeds = list(range(0, 10))

    pc_values = [f"{pc:.2f}" for pc in [0.92, 0.94, 0.96, 0.98]]
    theta_values = np.arange(4.0, 5.0 + 0.05, 0.1).tolist()

    output_mean, output_std = plotter.plot_ijcai_figure2_multiseed(
        data_dir=data_dir,
        pc_values=pc_values,
        theta_values=theta_values,
        seeds=seeds,
        output_prefix='ijcai_fig2_multiseed'
    )

    print(f"{output_mean=}")
    print(f"{output_std=}")


if __name__ == "__main__":
    main()
