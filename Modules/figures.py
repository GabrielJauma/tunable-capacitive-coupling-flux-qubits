import numpy as np
import matplotlib        as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# from palettable.cartocolors.sequential import BluGrn_7


# def animation(fig, plots, y_values, param_name, param_values):
#     def update(i):
#         for plot, y_value in zip(plots, y_values):
#             plot.set_ydata(y_value[i])
#         fig.suptitle(param_name + f'{param_values[i]}')
#         return plots
#
#     # Using FuncAnimation to create the animation
#     return FuncAnimation(fig, update, frames=len(param_values), blit=True)


def generate_colors_from_colormap(num_colors, colormap_name="Set3"):
    """
    Generate a list of colors from a specified colormap.

    Parameters:
    - num_colors (int): Number of colors to generate.
    - colormap_name (str): Name of the colormap.

    Returns:
    - list of colors
    """
    colormap = plt.cm.get_cmap(colormap_name, num_colors)
    return [mpl.colors.rgb2hex(colormap(i)) for i in range(colormap.N)]

def animation(fig, plots, y_values, param_name, param_values):
    def update(i):
        for plot, y_value in zip(plots, y_values):
            # If it's a line plot
            plot.set_ydata(y_value[i])
            
        fig.suptitle(param_name + f'{param_values[i]}')
        return plots

    # Using FuncAnimation to create the animation
    return FuncAnimation(fig, update, frames=len(param_values), blit=True)


def animation_contour(fig, plots, x, y, z_values, param_name, param_values):
    def update(i):
        for plot, z_value in zip(plots, z_values):
            ax = plot.axes  # Get the axes the contour belongs to

            # Remove old contours
            for collection in ax.collections:
                collection.remove()

            # Plot new contours using the same X, Y, and new Z values
            ax.contourf(x, y, z_value[i], cmap=plot.cmap)

        fig.suptitle(param_name + f'{param_values[i]}')
        return plots

    # Using FuncAnimation to create the animation
    return FuncAnimation(fig, update, frames=len(param_values), blit=True)


def multiline(c, ys, xs=None, fig= None, ax=None, cb=True, **kwargs):
    """Plot lines with different colorings

    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection

    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

    Returns
    -------
    lc : LineCollection instance.
    """

    ys = ys.T

    if xs is None:
        xs = np.tile(np.arange(np.shape(ys)[1]), (np.shape(ys)[0], 1))

    if np.shape(xs) != np.shape(ys):
        xs = np.tile(xs, (np.shape(ys)[0], 1))

    ax = plt.gca() if ax is None else ax
    fig = plt.gcf() if fig is None else fig

    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)
    lc.set_array(np.asarray(c))
    ax.add_collection(lc)
    ax.autoscale()
    if cb:
        fig.colorbar(lc, ax=ax)


def multiline_color_change(ys, colors, fig, ax, x=None, **kwargs):
    if x is None:
        x = np.arange(ys.shape[0] + 2)

    for i in range(ys.shape[1]):
        y = ys[:, i]
        y = np.append(y, np.array([-123456, -123456]))
        color = colors[:, i]
        color = np.append(color, np.array([colors.min(), colors.max()]))


        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, **kwargs)
        # Set the values used for colormapping
        lc.set_array(color)
        line = ax.add_collection(lc)
    fig.colorbar(line, ax=ax)
    ax.set_xlim(x.min(), x.max()-2)
    ax.set_ylim([ys.min() - abs(ys.min() * 0.1), ys.max() + abs(ys.min() * 0.1)])


def export(fig_title, fig=plt.gcf(), hspace=False, wspace=False):
    if hspace:
        fig.subplots_adjust(hspace=hspace)
    if wspace:
        fig.subplots_adjust(wspace=wspace)
    try:
        fig.savefig('Figures/' + fig_title, bbox_inches='tight')
    except:
        fig.savefig('../Figures/' + fig_title, bbox_inches='tight')


def transparent_cmap(cmap_name, power):
    'The output is a new colormap called cmap_name_t'
    # get colormap
    ncolors = 256
    color_array = plt.get_cmap(cmap_name)(range(ncolors))

    # change alpha values
    color_array[:, -1] = np.linspace(0, 1, ncolors) ** power

    # create a colormap object
    map_object = LinearSegmentedColormap.from_list(name=cmap_name + '_t', colors=color_array)

    # register this new colormap with matplotlib
    plt.register_cmap(cmap=map_object)

def colorbar_for_lines(fig, values, label=None, location='top'):
    norm = mpl.colors.Normalize(vmin=values[0], vmax=values[-1])
    sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
    sm.set_array([])

    if label is not None and location == 'top':
        a = fig.colorbar(sm, ticks=values)
        a.ax.set_title(label)

    elif label is not None and location == 'side':
        fig.colorbar(sm, ticks=values, label=label)

    else:
        fig.colorbar(sm, ticks=values)


def colorbar(mappable, ticks=False):
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    if ticks:
        cbar = fig.colorbar(mappable, cax=cax, ticks=ticks[0::int(len(ticks) / 10)])
    else:
        cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar
