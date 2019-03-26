############################################################################
#
#   A set of coordinate convert methods
#
#   DIZHONG.ZHU  28/June/2018
#
############################################################################
import numpy as np
from common import MatrixUtil


###########################################################
#   convert surface normal in cartesian to polar coordinates
#   @Input $ n: Nx3 unit normals
###########################################################
def normal2polar(n):
    zenith = np.arccos(n[..., 2])
    azimuth = np.arctan2(n[..., 1], n[..., 0])
    return zenith, azimuth


###########################################################
#   convert polar coordinate to unit surface normal
###########################################################
def polar2normal(zenith, azimuth):
    n = np.zeros((np.shape(zenith) + (3,)))
    n[..., 0] = np.sin(zenith) * np.cos(azimuth)
    n[..., 1] = np.sin(zenith) * np.sin(azimuth)
    n[..., 2] = np.cos(zenith)
    return n


def polar2pq1normal(zenith, azimuth):
    n = polar2normal(zenith, azimuth)
    n[..., 0] /= -n[..., 2]
    n[..., 1] /= -n[..., 2]
    n[..., 2] /= n[..., 2]
    return n


def depth2normal(z, mask, method='ff'):
    N = np.zeros((z.shape + (3,)), np.float32)

    Dx, Dy, newmask = MatrixUtil.GradientMatrix(mask, method)
    z_vec = z[mask == 1]
    p = Dx.dot(z_vec)
    q = Dy.dot(z_vec)

    norm = np.sqrt(p ** 2 + q ** 2 + 1)

    N[:, :, 0][newmask == 1] = -p / norm
    N[:, :, 1][newmask == 1] = -q / norm
    N[:, :, 2][newmask == 1] = 1 / norm

    # N = np.transpose(N, (1, 0, 2))

    return N, newmask


def depth2normal_perspective(z, fmask, varmask, Dx, Dy, P):
    z0 = z[varmask == 1]
    rows, cols = varmask.shape
    Z = np.ones((rows, cols)) * np.nan
    Z[varmask == 1] = z0
    c1 = P[0, 3] * P[1, 1] * P[2, 0]
    c2 = P[0, 1] * P[1, 3] * P[2, 0]
    c3 = P[0, 3] * P[1, 0] * P[2, 1]
    c4 = P[0, 0] * P[1, 3] * P[2, 1]
    c5 = P[0, 1] * P[1, 0] * P[2, 3]
    c6 = P[0, 0] * P[1, 1] * P[2, 3]
    C1 = c1 - c2 - c3 + c4 + c5 - c6

    c7 = P[0, 2] * P[1, 1] * P[2, 0]
    c8 = P[0, 1] * P[1, 2] * P[2, 0]
    c9 = P[0, 2] * P[1, 0] * P[2, 1]
    c10 = P[0, 0] * P[1, 2] * P[2, 1]
    c11 = P[0, 1] * P[1, 0] * P[2, 2]
    c12 = P[0, 0] * P[1, 1] * P[2, 2]
    C2 = c10 + c11 - c12 + c7 - c8 - c9

    x, y = np.meshgrid(np.linspace(0, cols - 1, cols), np.linspace(0, rows - 1, rows))
    denorm = -P[0, 0] * P[1, 1] + P[1, 1] * P[2, 0] * x + P[0, 0] * P[2, 1] * y + P[0, 1] * (P[1, 0] - P[2, 0] * y)

    Zx = np.ones((rows, cols)) * np.nan
    Zy = np.ones((rows, cols)) * np.nan
    Zx[fmask == 1] = Dx.dot(z0)
    Zy[fmask == 1] = Dy.dot(z0)

    pnx = -((-P[0, 0] + P[2, 0] * x) * Zx + (-P[1, 0] + P[2, 0] * y) * Zy) * (C1 + C2 * Z) / denorm ** 2
    pny = ((P[0, 1] - P[2, 1] * x) * Zx + (P[1, 1] - P[2, 1] * y) * Zy) * (C1 + C2 * Z) / denorm ** 2
    pnz = -(C1 + denorm * ((-P[0, 2] + P[2, 2] * x) * Zx + (-P[1, 2] + P[2, 2] * y) * Zy) + C2 * Z) * (
            C1 + C2 * Z) / denorm ** 3

    norm = np.sqrt(pnx ** 2 + pny ** 2 + pnz ** 2)

    PN = np.ones((rows, cols, 3)) * np.nan

    PN[..., 0] = pnx / norm
    PN[..., 1] = pny / norm
    PN[..., 2] = pnz / norm
    return PN
