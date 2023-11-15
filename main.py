"""
A small showcase of the functionality
"""
from src import distribution
from mpl import kit_colors


if __name__ == "__main__":
    # create a new log normal distribution
    my_dist = distribution.get_distribution(kind='log-normal', x_lim=[1e-1, 1e2], base=3, x_type=1, nodes=60)
    # plot population distribution
    fig, ax = my_dist.plot()
    fig.show()

    # plot cumulative distribution
    fig_c, ax_c = my_dist.plot(mode=1)
    fig_c.show()

    # convert distribution
    my_dist.change_base_and_x_type(base=0, x_type=3).plot_on_ax(c=kit_colors.red, ax=ax)
    my_dist.change_base_and_x_type(base=0, x_type=3).plot_on_ax(c=kit_colors.red, ax=ax_c, mode=1)

    # change dicretization
    my_dist.discretize(x_max=[1e-2, 2e1], nodes=30).plot_on_ax(c=kit_colors.green, ax=ax)
    my_dist.discretize(x_max=[1e-2, 2e1], nodes=30).plot_on_ax(c=kit_colors.green, ax=ax_c, mode=1)

    # mix distributions
    my_dist_2 = distribution.get_distribution(kind='log-normal', x_lim=[1e-1, 1e2], base=3, x_type=1, nodes=60, mu=7)
    fig_mix, ax_mix = my_dist_2.plot()
    fig_mix.show()

    mixed_dist = my_dist.mix(my_dist_2, mass_frac=[0.6])
    mixed_dist.plot_on_ax(ax=ax_mix, c=kit_colors.cyan)
