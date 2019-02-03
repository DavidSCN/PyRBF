""" Testfunctions, should be evaluated on [0,1]. """

import numpy as np

class Testfunction:
    def __str__(self):
        return type(self).__name__
    

class Highfreq(Testfunction):
    def __call__(self, x):
        return np.exp(3*x) + 4*np.sin(5*x) + np.sin(80*x) + 2


class Lowfreq(Testfunction):
    def __call__(self, x):
        return np.exp(3*x) + 4*np.sin(5*x) + 2

class Jump(Testfunction):
    def __call__(self, x):
        return np.piecewise(x,
                            [x <= 0.5, x > 0.5 ],
                            [2,        4])

class Constant(Testfunction):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return "Constant" + str(self.value)

    def __call__(self, x):
        return np.full_like(x, self.value)



# For 3d function coords is expected to be an n x 2 array.
# [ [x1, y1], [x2, y2], [x3, y3], ...]

def eggholder(coords):
    """ According to https://en.wikipedia.org/wiki/Test_functions_for_optimization
    Eggholder function shifted to [0,1]x[0,1]. """
    x = (coords[:,0] - 0.5) * 512
    y = (coords[:,1] - 0.5) * 512
        
    return -(y+47) * np.sin(np.sqrt(np.abs(x/2 + y+47))) - x*np.sin(np.sqrt(np.abs(x-(y+47)))) + 400


def rosenbrock(coords):
    """ https://en.wikipedia.org/wiki/Rosenbrock_function. Shifted from [-2,2] to [0,1] """
    a = 1
    b = 100
    x = (coords[:,0] - 0.5) * 4
    y = (coords[:,1] - 0.5) * 4
    return np.square(a-x) + b * np.square((y-np.square(x)))
    # return scipy.optimize.rosen(coords)


def platform(coords):
    """ A jump between 4 different niveaus. """
    vectors = np.piecewise(coords,
                           [
                               (coords[:,0] <= 0.5) & (coords[:,1] <= 0.5), # lower right
                               (coords[:,0] <= 0.5) & (coords[:,1] > 0.5), # lower left
                               (coords[:,0] > 0.5) & (coords[:,1] > 0.5), # upper left
                               (coords[:,0] > 0.5) & (coords[:,1] <= 0.5)  # upper right
                           ],
                           [ 2, 5, 7, 8 ]
    )
    return vectors[:,0]


import matplotlib.pyplot as plt

if __name__ == "__main__":
    x = np.linspace(0, 1, 50)
    y = np.linspace(0, 1, 50)

    grid = np.meshgrid(x, y)
    coords = np.vstack(np.meshgrid(x,y)).reshape(2, -1).T
    z = platform(coords)
    print(z)
    plt.contourf(x, y, z.reshape(50,50))
    plt.legend()
    plt.show()
    set_trace()
    
    
