import numpy as np
import math


def translate_scale_hyperrectangle(x, x_0, size):
    scale = [math.sqrt(2) / size_dim for size_dim in size]
    translate = [-x_0_dim for x_0_dim in x_0]
    return translate_scale_with(x, translate, scale), translate, scale

def translate_scale_with(x, translate, scale):
    assert (x.shape[0] == len(scale) == len(translate))
    dim = len(scale)
    unit_x = np.empty(x.shape)
    for i in range(dim):
        unit_x[i, :] = (x[i, :] + translate[i]) * scale[i]
    assert (np.all(np.abs(unit_x) <= math.sqrt(2) / 2 + 1e7))
    return unit_x

def get_center_extents(mesh):
    dim = mesh.shape[0]
    xmax = [mesh[i, :].max() for i in range(dim)]
    xmin = [mesh[i, :].min() for i in range(dim)]
    extents = [xmax[i] - xmin[i] for i in range(dim)]
    center = [xmin[i] + extents[i] / 2 for i in range(dim)]
    return center, extents