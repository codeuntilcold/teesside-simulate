#!/usr/bin/env python3
"""
Generate NEB Figure 1 style time series grid.
Similar to IJCAI Figure 1 but for NEB (Neighborhood-based) strategy.
"""

from plotting_utils import SimulationPlotter


def main():
    plotter = SimulationPlotter()
    data_dir = 'data/neb_theta'

    # NEB parameters
    nc_values = [1, 2, 3, 4]  # Neighborhood cooperator thresholds
    theta = 5.5  # Theta value to visualize

    print("=" * 70)
    print("NEB Figure 1 Grid Generation")
    print("=" * 70)
    print(f"\nGenerating grid plot:")
    print(f"  NC values: {', '.join(map(str, nc_values))}")
    print(f"  Theta: {theta}")
    print(f"  Layout: 3 rows (frequency, cost, social welfare) × {len(nc_values)} columns")

    filename = plotter.plot_neb_figure1_grid(
        data_dir=data_dir,
        nc_values=nc_values,
        theta_value=theta,
        show_plot=False
    )

    print(f"\n{'=' * 70}")
    print("Grid generation complete!")
    print(f"{'=' * 70}")
    print(f"\nOutput: {filename}")
    print("\nThis layout matches IJCAI Figure 1 (NEB row) but with social welfare added!")


if __name__ == "__main__":
    main()
