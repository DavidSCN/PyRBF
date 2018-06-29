import numpy as np
import math


def translate_scale_hyperrectangle(x, x_0, size):
    """
    Perform translation (to origin) and then scaling on a hyperrectangle. This is used
    to translate a mesh within a hyperrectangle to the origin and scale it to
    the interval [-1, 1] in 1-D, and for higher dimensions analogously.
    :param x: the mesh to be transformed
    :param x_0: the center of the hyperrectangle
    :param size: the size of the rectangle in each dimension
    :return: the transformed mesh and the used tranlslation and scale vectors.
    """
    scale = [2 * math.sqrt(len(size)) / (size_dim * len(size)) for size_dim in size]
    translate = [-x_0_dim for x_0_dim in x_0]
    return translate_scale_with(x, translate, scale), translate, scale


def translate_scale_with(x, translate, scale):
    """
    Perform abitrary translation and scaling on a mesh (in this order).
    :param x: the mesh
    :param translate: translation vector
    :param scale: scale vector
    :return: the transformed mesh
    """
    assert (x.shape[0] == len(scale) == len(translate))
    dim = len(scale)
    unit_x = np.empty(x.shape)
    for i in range(dim):
        unit_x[i, :] = (x[i, :] + translate[i]) * scale[i]
    assert (np.all(np.abs(unit_x) <= math.sqrt(2) / 2 + 1e7))
    return unit_x


def get_center_extents(mesh):
    """
    For a given set of points this returns a hyperrectangle that contains all points
    :param mesh: the mesh
    :return: the hyperrectangle, given through a center and extents in each dimension
    """
    dim = mesh.shape[0]
    xmax = [mesh[i, :].max() for i in range(dim)]
    xmin = [mesh[i, :].min() for i in range(dim)]
    extents = [xmax[i] - xmin[i] for i in range(dim)]
    center = [xmin[i] + extents[i] / 2 for i in range(dim)]
    return center, extents


def cart2polar(mesh):
    """
    This performs coordinate transformation in 1,2 and 3-D from cartesian
    coordinates to polar (spherical) coordinates. Convention ISO 80000-2:2009
    is used for spherical coordinates.
    Note that this will copy the mesh even for the trivial 1-D case.
    :param mesh: the mesh
    :return: the mesh in polar (spherical) coordinates
    """
    polarmesh = np.empty(mesh.shape)
    if mesh.shape[0] == 1:
        polarmesh[0, :] = mesh[0, :]
    elif mesh.shape[0] == 2:
        polarmesh[0, :] = np.sqrt(mesh[0, :]**2 + mesh[1, :]**2)
        polarmesh[1, :] = np.arctan2(mesh[1, :], mesh[0, :])
    elif mesh.shape[0] == 3:
        polarmesh[0, :] = np.sqrt(mesh[0, :]**2 + mesh[1, :]**2 + mesh[2, :]**2)
        polarmesh[1, :] = np.arccos(np.nan_to_num(mesh[2, :] / polarmesh[0, :]))
        polarmesh[2, :] = np.arctan2(mesh[1, :], mesh[0, :])
    else:
        raise AssertionError("Mesh has invalid number of dimensions: " + str(mesh.shape[0]))
    return polarmesh