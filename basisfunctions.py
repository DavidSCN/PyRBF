import functools
import numpy as np
import scipy, scipy.special

class Basisfunction():
    has_shape_param = False

    def __init__(self, shape_parameter = None):
        self.s = shape_parameter
    
    def __str__(self):
        return type(self).__name__

    @classmethod
    def shape_param_from_m(cls, m, in_mesh):
        """ m is the support radius expressed in multiples of the mesh width.
        Default implementation for basis function that don't have a shape parameter or
        the parameter can not be interpreted that way. """
        return m

    def shaped(self, m, in_mesh):
        return functools.partial(self.__call__, shape = self.shape_param(m, in_mesh))

    @staticmethod
    def h_max(mesh):
        """ Finds the greatest distance to each vertices nearest neighbor. """
        h_max = 0
        if mesh.ndim == 1:
            mesh = mesh[:, np.newaxis]
            
        for i in mesh:
            distances = np.linalg.norm(mesh - i, axis=1)
            h_max = np.max([ np.min(distances[distances != 0]), h_max])

        return h_max


class Gaussian(Basisfunction):
    has_shape_param = True

    def __init__(self, shape_parameter):
        self.s = shape_parameter
      
    def __call__(self, radius):
        radius = np.atleast_1d(radius)
        threshold = np.sqrt( - np.log(10e-9) ) / self.s
        result = np.exp( -np.power(self.s * np.abs(radius), 2))
        result[ radius > threshold ] = 0;
        return result

    @classmethod
    def shape_param_from_m(cls, m, in_mesh):
        h_max = cls.h_max(in_mesh)
        return np.sqrt(-np.log(1e-9)) / (m*h_max)
            

class ThinPlateSplines(Basisfunction):
    def __call__(self, radius):
        """ Thin Plate Splines Basis Function """
        # Avoids the division by zero in np.log
        return scipy.special.xlogy(np.power(radius, 2), np.abs(radius))

    
class InverseMultiQuadrics(Basisfunction):
    has_shape_param = True
    
    def __init__(self, shape_parameter):
        self.s = shape_parameter
    
    def __call__(self, radius):
        return 1.0 / np.sqrt(np.power(self.s, 2) + np.power(radius, 2));

    
class MultiQuadrics(Basisfunction):
    has_shape_param = True

    def __init__(self, shape_parameter):
        self.s = shape_parameter
    
    def __call__(self, radius):
        return np.sqrt(np.power(self.s, 2) + np.power(radius, 2))

    
class VolumeSplines(Basisfunction):
    def __call__(self, radius):
        return np.abs(radius)

    
class CompactThinPlateSplineC2(Basisfunction):
    has_shape_param = True

    def __init__(self, shape_parameter):
        self.s = shape_parameter
    
    def __call__(self, radius):
        radius = np.abs(radius)
        result = np.zeros_like(radius)
        p = radius / self.s
        result =  1 - 30*np.power(p, 2) - 10*np.power(p, 3) + 45*np.power(p, 4) - 6*np.power(p, 5)
        result =- scipy.special.xlogy(60*np.power(p,3), p)
        result[ radius >= self.s ] = 0
        return result

    @classmethod
    def shape_param_from_m(cls, m, in_mesh):
        h_max = cls.h_max(in_mesh)
        return m * h_max


class CompactPolynomialC0(Basisfunction):
    has_shape_param = True

    def __init__(self, shape_parameter):
        self.s = shape_parameter
    
    def __call__(self, radius):
        radius = np.abs(radius)
        result = np.zeros_like(radius)
        p = radius / self.s
        result =  np.power(1-p, 2)
        result[ radius >= self.s ] = 0
        return result

    @classmethod
    def shape_param_from_m(cls, m, in_mesh):
        h_max = cls.h_max(in_mesh)
        return  m * h_max


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
