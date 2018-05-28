import functools
import numpy as np
import scipy, scipy.special

class Basisfunction():
    def shape_param(self, m, in_mesh):
        return m
    
    def shaped(self, m, in_mesh):
        return functools.partial(self.__call__, shape = self.shape_param(m, in_mesh))

    def h_max(self, mesh):
        """ Find the greatest distance to each vertices nearest neighbor. """
        h_max = 0
        if mesh.ndim == 1:
            mesh = mesh[:, np.newaxis]
            
        for i in mesh:
            distances = np.linalg.norm(mesh - i, axis=1)
            h_max = np.max([ np.min(distances[distances != 0]), h_max])

        return h_max


class Gaussian(Basisfunction):
    def __call__(self, radius, shape):
        radius = np.atleast_1d(radius)
        threshold = np.sqrt( - np.log(10e-9) ) / shape
        result = np.exp( -np.power(shape*np.abs(radius), 2))
        result[ radius > threshold ] = 0;
        return result

    def shape_param(self, m, in_mesh):
        h_max = self.h_max(in_mesh)
        return np.sqrt(-np.log(1e-9)) / (m*h_max)
            

class ThinPlateSplines(Basisfunction):
    def __call__(self, radius, shape = 0):
        """ Thin Plate Splines Basis Function """
        # Avoids the division by zero in np.log
        return scipy.special.xlogy(np.power(radius, 2), np.abs(radius))

class InverseMultiQuadrics(Basisfunction):
    def __call__(self, radius, shape):
        return 1.0 / np.sqrt(1 + np.power(shape, 2) + np.power(radius, 2));

class MultiQuadrics(Basisfunction):
    def __call__(self, radius, shape):
        return np.power(shape, 2) + np.power(radius, 2)

class VolumeSplines(Basisfunction):
    def __call__(self, radius, shape = 0):
        return np.abs(radius)

class CompactThinPlateSplineC2(Basisfunction):
    def __call__(self, radius, shape):
        radius = np.abs(radius)
        result = np.zeros_like(radius)
        p = radius / shape
        result =  1 - 30*np.power(p, 2) - 10*np.power(p, 3) + 45*np.power(p, 4) - 6*np.power(p, 5) - 60*np.log(np.power(p, np.power(p, 3)))
        result[ radius >= shape ] = 0
        return result


def rescaleBasisfunction(func, m, in_mesh):
    """ Returns the basis function shape parameter, so that it has decayed to < 10^9 at x = m*h. h being the maximum cell width. Assumes in_mesh being sorted."""
    try:
        h = np.max(in_mesh[1:] - in_mesh[:-1])
    except IndexError:
        h = in_mesh
        
    if func == Gaussian:
        s = np.sqrt(-np.log(1e-9)) / (m*h)
    elif func == ThinPlateSPlines:
        return 0 # TPS has no shape parameter
    else:
        raise NotImplemented        
#    print("Maximum input mesh distance h =", h, " resulting in shape parameter =", s)
    return s