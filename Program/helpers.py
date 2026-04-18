import numpy as np

"""
Helper function to rotate image.
This is only used for testing to see whether the rotation of the image impacts our matching
"""
def rotate_image(img, angle, cx, cy):
    if img.ndim < 2:
        return img

    k = int(round(float(angle) / 90.0))
    if abs(float(angle) - 90.0 * k) > 1e-9:
        raise ValueError("rotateImage only supports multiples of 90 degrees without intensity interpolation")

    k %= 4
    if k == 0:
        return img

    h, w = img.shape[:2]
    y, x = np.indices((h, w))
    dx = x - float(cx)
    dy = y - float(cy)

    if k == 1:
        x_new = float(cx) - dy
        y_new = float(cy) + dx
    elif k == 2:
        x_new = float(cx) - dx
        y_new = float(cy) - dy
    else:
        x_new = float(cx) + dy
        y_new = float(cy) - dx

    x_new = np.rint(x_new).astype(int)
    y_new = np.rint(y_new).astype(int)

    out = img.copy()
    keep = (x_new >= 0) & (x_new < w) & (y_new >= 0) & (y_new < h)
    out[y_new[keep], x_new[keep]] = img[y[keep], x[keep]]
    return out
