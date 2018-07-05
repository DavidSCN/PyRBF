import matplotlib
import math
import scipy.constants

def multi_legend(ax1, ax2, loc):
    """ Combines the labels of two axes to one legend. """
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    ax2.legend(lines + lines2, labels + labels2,
               loc = loc)
                
    # leg.set_alpha(1)
    # leg.get_frame().set_facecolor("w")
    

def set_save_fig_params(fig_width_pt = 418.25555, rows = 1, cols = 1):
    """ From: http://scipy-cookbook.readthedocs.io/items/Matplotlib_LaTeX_Examples.html """
    # Get this from LaTeX using \showthe\columnwidth
    # fig_width_pt = 418.25555                # Got this from LaTeX for scrreprt, scrartcl using \showthe\columnwidth
    # fig_width_pt = 307.0                    # from Beamer Uni Stuttgart Theme
    # matplotlib.style.use('seaborn')
    small = (rows <= 1)
    inches_per_pt = 1.0 / 72.0                # Convert pt to inches
    golden_mean = scipy.constants.golden_ratio - 1  # Aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt  # width in inches
    fig_height = fig_width * golden_mean      # height in inches
    fig_size = [fig_width * cols, fig_height * rows]

    font_offset = 0
    
    params = {'backend': 'pdf',
              'axes.titlesize': 10,
              'axes.labelsize': 8 - font_offset,
              'font.size': 10 - font_offset,
              'legend.fontsize': 7 - font_offset,
              'legend.framealpha' : 1,
              # 'legend.fancybox' : True,
              'xtick.labelsize': 7 - font_offset,
              'ytick.labelsize': 7 - font_offset,
              'lines.linewidth' : 1,
              'text.usetex': True,
              'figure.figsize': fig_size,
              "pgf.rcfonts" : False}
    matplotlib.rcParams.update(params)
