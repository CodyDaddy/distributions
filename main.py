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
    my_dist.plot_on_ax(ax=ax, mode=1, c=kit_colors.green)

    # convert distribution
    my_dist.change_base_and_x_type(base=0, x_type=3).plot_on_ax(c=kit_colors.red, ax=ax, mode=1)

    # change discretization
    ax.clear()
    my_dist.plot_on_ax(ax=ax, c=kit_colors.green, label=f'{len(my_dist.x)} nodes')
    my_dist.discretize(x_max=[1e-2, 2e1], nodes=30).plot_on_ax(c=kit_colors.red, ax=ax, label=f'30 nodes')

    # mix two distributions
    ax.clear()
    my_dist.plot_on_ax(ax=ax, label='my_dist')
    my_dist_2 = distribution.get_distribution(kind='log-normal', x_lim=[1e-1, 1e2], base=3, x_type=1, nodes=60, mu=7)
    my_dist_2.plot_on_ax(ax=ax, c=kit_colors.red, label='my_dist_2')

    mixed_dist = my_dist.mix(my_dist_2, mass_frac=[0.6])
    mixed_dist.plot_on_ax(ax=ax, c=kit_colors.cyan, label='my_dist_mix')

    ax.clear()
    my_dist.plot_on_ax(ax=ax, label='my_dist')
    my_dist_2.plot_on_ax(ax=ax, c=kit_colors.red, label='my_dist_2')
    my_dist_3 = distribution.get_distribution(kind='log-normal', x_lim=[1e-1, 1e2], base=3, x_type=1, nodes=60, mu=3)
    my_dist_3.plot_on_ax(ax=ax, c=kit_colors.orange, label='my_dist_2')

    # mix 3 distributions
    mixed_dist = my_dist.mix(my_dist_2, my_dist_3, mass_frac=[0.3, 0.25])
    mixed_dist.plot_on_ax(ax=ax, c=kit_colors.cyan, label='my_dist_mix')

    # calculate moments
    ax.clear()
    # total mass
    print('total mass: ', my_dist.get_moment())
    # average particle size (based on mass)
    print('average particle size (based on mass): ', my_dist.get_moment(1))
    # arithmetic average particle size (based on number)
    print('arithmetic average particle size (based on number): ', my_dist.change_base(0).get_moment(1))

    # arithmetic average particle size (based on number)
    mixed_dist.plot_on_ax(ax=ax, mode=1, c=kit_colors.green, lw=3, marker=None)
    ax.hlines(y=.4, xmin=.1, xmax=100, linewidth=1, color=kit_colors.red)
    ax.vlines(x=2.32, ymin=0, ymax=1, linewidth=1, color=kit_colors.red)
    print('percentile of value x = 2.32: ', mixed_dist.get_percentile_of_x(2.32))
    print('value x_40: ', mixed_dist.get_x_of_percentile(.4))

    print('end of show case')
