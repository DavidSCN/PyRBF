from rbf import *
from RBF.basisfunctions import *
import matplotlib.pyplot as plt

def basisfunctions_survey():
    in_mesh = np.linspace(1, 4, 200)
    plot_mesh = np.linspace(1, 4, 5000)
    in_vals = func(in_mesh)
    
    plot_range = np.linspace(-3, 3, 1000)
    range_max = 3
    BFs = [ {"func" : Gaussian().shaped(14, in_mesh),             
             "name" : "Gaussian",
             "pr"   : np.linspace(-0.5, 0.5, 1000),
             "title": "$\phi(|x|) = \exp(-(s \cdot x)^2)$"},

            {"func" : functools.partial(VolumeSplines(), shape = 1),
             "name" : "Volume Splines",
             "pr"   : np.linspace(-range_max, range_max, 1000),
             "title" : "$\phi(|x|) = x$"},
            
            {"func" : ThinPlateSplines(),
             "name" : "Thin Plate Splines",
             "pr"   : np.linspace(-range_max, range_max, 1000),
             "title": "$\phi(|x|) = \log(x) \cdot x^2$"},
            
            {"func" : functools.partial(MultiQuadrics(), shape = 0.001),
             "name" : "Multi Quadrics",
             "pr"   : np.linspace(-range_max, range_max, 1000),
             "title": "$\phi(|x|) = s^2 + x^2$"},
            
            {"func" : functools.partial(InverseMultiQuadrics(), shape = 0.1),
             "name" : "Inverse Multi Quadrics",
             "pr"   : np.linspace(-range_max, range_max, 1000),
             "title": "$\phi(|x|) = 1 / (s^2 + x^2)$"}            
            ]

    ax1 = plt.subplot2grid((3,5), (0,0), colspan = 5)
    ax2 = plt.subplot2grid((3,5), (1,0), colspan = 5)
    rmse = []
    cond = []
    names = []
    width = 0.5
    for i, bf in enumerate(BFs):
        sep_consistent = SeparatedConsistent(bf["func"], in_mesh, in_vals, rescale = False)
        rmse.append(sep_consistent.RMSE(func, plot_mesh))
        names.append(bf["name"])
        cond.append(sep_consistent.condC)
        ax = plt.subplot2grid((3,5), (2,i))
        ax.plot(bf["pr"], bf["func"](bf["pr"]))
        ax.set_xlabel(bf["title"])
        print(bf["name"], "RMSE =", rmse[-1], "Condition =", cond[-1])
        
    index = np.arange(len(rmse))
    ax1.bar(index, rmse, width, label = "RMSE", log=True)
    ax1.set_ylabel("RMSE")
    ax1.set_xticks([])

    ax2.bar(index, cond, width, label = "Condition", log=True)
    ax2.set_ylabel("Condition")
    ax2.set_xticks(index)
    ax2.set_xticklabels(names)

    
    # ax[3].subplots(1, 5)

    # plt.legend()
    plt.show()
        
            
    
basisfunctions_survey()
