import numpy as np
import scipy.optimize

def testfunction_1d(x):
    return np.exp(2*x) + 4*np.sin(5*x) + np.sin(80*x) + 10


def eggholder(coords):
    """ According to https://en.wikipedia.org/wiki/Test_functions_for_optimization
    Eggholder function shifted to [0,1]x[0,1]. """
    x = (coords[:,0] - 0.5) * 512
    y = (coords[:,1] - 0.5) * 512
    
    return -(y+47) * np.sin(np.sqrt(np.abs(x/2 + y+47))) - x*np.sin(np.sqrt(np.abs(x-(y+47))))


def rosenbrock(coords):
    """ https://en.wikipedia.org/wiki/Rosenbrock_function. Shifted from [-2,2] to [0,1] """
    a = 1
    b = 100
    x = (coords[:,0] - 0.5) * 4
    y = (coords[:,1] - 0.5) * 4
    return np.square(a-x) + b * np.square((y-np.square(x)))
    # return scipy.optimize.rosen(coords)
