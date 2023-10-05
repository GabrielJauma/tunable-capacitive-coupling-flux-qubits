import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable


def export(fig_title):
    plt.gcf()
    plt.margins(0,0)
    plt.savefig('Figures/'  + fig_title, bbox_inches = 'tight', pad_inches = 0.05)
    
def transparent_cmap(cmap_name, power): 
    'The output is a new colormap called cmap_name_t'
    # get colormap
    ncolors = 256
    color_array = plt.get_cmap(cmap_name)(range(ncolors))

    # change alpha values
    color_array[:,-1] = np.linspace(0,1,ncolors)**power

    # create a colormap object
    map_object = LinearSegmentedColormap.from_list(name=cmap_name + '_t',colors=color_array)

    # register this new colormap with matplotlib
    plt.register_cmap(cmap=map_object)
    
def colorbar(mappable, ticks):
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax, ticks= ticks)
    plt.sca(last_axes)
    return cbar