############################################################################
#
#   Matrices helpers for building different sparse matrix
#
#   DIZHONG.ZHU  02/Nov/2018
#
############################################################################
from scipy import sparse
import numba
import numpy as np


# Build a rectangle identity sparse matrix
# The mask contains less or equal elements then basemask
def rect_I(mask, basemask):
    rows, cols = mask.shape
    offsetRow = maskoffset(mask)
    offsetCol = maskoffset(basemask)
    noofRowPixels = np.sum(mask)
    noofColPixels = np.sum(basemask)
    count = 0

    idx_v = np.zeros(noofRowPixels)
    idx_f = np.zeros(noofRowPixels)
    idx_value = np.ones(noofRowPixels)

    for m in range(rows):
        for n in range(cols):
            if mask[m, n] == 1:
                row_idx = m * cols + n - offsetRow[m, n]
                col_idx = m * cols + n - offsetCol[m, n]
                idx_v[count] = col_idx
                idx_f[count] = row_idx
                count += 1

    idx_v = idx_v[0:count]
    idx_f = idx_f[0:count]
    idx_value = idx_value[0:count]
    return sparse.coo_matrix((idx_value, (idx_f, idx_v)), shape=(noofRowPixels, noofColPixels))


@numba.jit
def maskoffset(mask):
    rows, cols = mask.shape
    offm = np.zeros((rows, cols), np.int32)
    maskCount = 0

    for m in range(rows):
        for n in range(cols):
            if mask[m, n] == 0:
                maskCount += 1
            else:
                offm[m, n] = maskCount
    return offm


#####################################################################
#   Construct a sparse gradient matrix for valid pixels specify by
#   mask
#   @Input
#       % mask: a MxN mask, where the valid pixels equal to one
#       % method: specify the finite different method
#####################################################################
def GradientMatrix(mask, method='ff'):
    offsetm = maskoffset(mask)

    mask_pad = np.pad(mask, ((1, 1), (1, 1)), 'constant', constant_values=0)
    offsetm = np.pad(offsetm, ((1, 1), (1, 1)), 'constant', constant_values=0)

    rows, cols = mask_pad.shape

    validPixel_mask = np.zeros_like(mask_pad)

    # 1. Get a mask for valid pixels
    if method == 'bb':
        for m in range(rows):
            for n in range(cols):
                if mask_pad[m, n] and mask_pad[m, n - 1] and mask_pad[m - 1, n]:
                    validPixel_mask[m, n] = 1
    elif method == 'bf':
        for m in range(rows):
            for n in range(cols):
                if mask_pad[m, n] and mask_pad[m, n - 1] and mask_pad[m + 1, n]:
                    validPixel_mask[m, n] = 1
    elif method == 'ff':
        for m in range(rows):
            for n in range(cols):
                if mask_pad[m, n] and mask_pad[m, n + 1] and mask_pad[m + 1, n]:
                    validPixel_mask[m, n] = 1
    elif method == 'fb':
        for m in range(rows):
            for n in range(cols):
                if mask_pad[m, n] and mask_pad[m, n + 1] and mask_pad[m - 1, n]:
                    validPixel_mask[m, n] = 1

    # 2. Update the mask
    newmask_pad = mask_pad & validPixel_mask
    newmask = newmask_pad[1:rows - 1, 1:cols - 1]
    offsetnm = maskoffset(newmask)
    offsetnm = np.pad(offsetnm, ((1, 1), (1, 1)), 'constant', constant_values=0)

    noofVariables = np.sum(mask)
    noofFunctions = np.sum(newmask)

    Dx_idx_val = np.zeros((noofVariables * 2, 3))
    Dy_idx_val = np.zeros((noofVariables * 2, 3))
    Dx_cout = 0
    Dy_cout = 0

    # construct the Gradient matrix index
    for m in range(rows):
        for n in range(cols):
            if validPixel_mask[m, n]:
                f_idx = (cols - 2) * (m - 1) + n - 1 - offsetnm[m, n]
                v_idx = (cols - 2) * (m - 1) + n - 1 - offsetm[m, n]
                Dx_idx_val[Dx_cout, :] = (f_idx, v_idx, -1)
                Dx_cout += 1
                Dy_idx_val[Dy_cout, :] = (f_idx, v_idx, -1)
                Dy_cout += 1

                if method == 'bb':
                    v_idx = (cols - 2) * (m - 1) + n - 2 - offsetm[m, n - 1]
                    Dx_idx_val[Dx_cout, :] = (f_idx, v_idx, 1)
                    Dx_cout += 1

                    v_idx = (rows - 2) * (m - 2) + n - 1 - offsetm[m - 1, n]
                    Dy_idx_val[Dy_cout, :] = (f_idx, v_idx, 1)
                    Dy_cout += 1
                elif method == 'bf':
                    v_idx = (cols - 2) * (m - 1) + n - 2 - offsetm[m, n - 1]
                    Dx_idx_val[Dx_cout, :] = (f_idx, v_idx, 1)
                    Dx_cout += 1

                    v_idx = (cols - 2) * m + n - 1 - offsetm[m + 1, n]
                    Dy_idx_val[Dy_cout, :] = (f_idx, v_idx, 1)
                    Dy_cout += 1
                elif method == 'ff':
                    v_idx = (cols - 2) * (m - 1) + n - offsetm[m, n + 1]
                    Dx_idx_val[Dx_cout, :] = (f_idx, v_idx, 1)
                    Dx_cout += 1

                    v_idx = (cols - 2) * m + n - 1 - offsetm[m + 1, n]
                    Dy_idx_val[Dy_cout, :] = (f_idx, v_idx, 1)
                    Dy_cout += 1
                elif method == 'fb':
                    v_idx = (cols - 2) * (m - 1) + n - offsetm[m, n + 1]
                    Dx_idx_val[Dx_cout, :] = (f_idx, v_idx, 1)
                    Dx_cout += 1

                    v_idx = (cols - 2) * (m - 2) + n - 1 - offsetm[m - 1, n]
                    Dy_idx_val[Dy_cout, :] = (f_idx, v_idx, 1)
                    Dy_cout += 1

    Dx_idx_val = Dx_idx_val[0:Dx_cout]
    Dy_idx_val = Dy_idx_val[0:Dy_cout]
    Dx = sparse.coo_matrix((Dx_idx_val[:, 2], (Dx_idx_val[:, 0], Dx_idx_val[:, 1])),
                           shape=(noofFunctions, noofVariables))
    Dy = sparse.coo_matrix((Dy_idx_val[:, 2], (Dy_idx_val[:, 0], Dy_idx_val[:, 1])),
                           shape=(noofFunctions, noofVariables))

    return Dx, Dy, newmask


#####################################################################
#   Construct a sparse Laplacian matrix for valid pixels specify by
#   mask, where the mask should have 4 neighbours
#   @Input
#       % mask: a MxN mask, where the valid pixels equal to one
#####################################################################
def LaplacianSparseMatrix(mask):
    rows, cols = mask.shape
    newmask = np.zeros_like(mask)

    offsetm = maskoffset(mask)
    noofValidPixels = np.sum(mask)

    count = 0
    idx_v = np.zeros(noofValidPixels * 5)
    idx_f = np.zeros(noofValidPixels * 5)
    idx_value = np.ones(noofValidPixels * 5)

    for m in range(rows):
        for n in range(cols):
            if mask[m, n]:
                if n - 1 >= 0 and n + 1 < cols and m - 1 >= 0 and m + 1 < rows:
                    if mask[m, n - 1] and mask[m, n + 1] and mask[m - 1, n] and mask[m + 1, n]:
                        newmask[m, n] = 1

    offsetnm = maskoffset(newmask)
    noofFunctions = np.sum(newmask)

    for m in range(rows):
        for n in range(cols):
            if newmask[m, n]:
                fidx = m * cols + n - offsetnm[m, n]

                idx_f[count] = fidx
                idx_v[count] = m * cols + n - offsetm[m, n]
                idx_value[count] = -4
                count += 1

                idx_f[count] = fidx
                idx_v[count] = m * cols + n - 1 - offsetm[m, n - 1]
                idx_value[count] = 1
                count += 1

                idx_f[count] = fidx
                idx_v[count] = m * cols + n + 1 - offsetm[m, n + 1]
                idx_value[count] = 1
                count += 1

                idx_f[count] = fidx
                idx_v[count] = (m - 1) * cols + n - offsetm[m - 1, n]
                idx_value[count] = 1
                count += 1

                idx_f[count] = fidx
                idx_v[count] = (m + 1) * cols + n - offsetm[m + 1, n]
                idx_value[count] = 1
                count += 1

    idx_f = idx_f[0:count]
    idx_v = idx_v[0:count]
    idx_value = idx_value[0:count]
    L = sparse.coo_matrix((idx_value, (idx_f, idx_v)), shape=(noofFunctions, noofValidPixels))

    return L, newmask
