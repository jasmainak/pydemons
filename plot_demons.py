import numpy as np
import matplotlib.pyplot as plt
from pydemons import demons, iminterpolate
from PIL import Image

moving = np.array(Image.open("mdb001.pgm"), dtype=np.float)
fixed = np.array(Image.open("mdb003.pgm"), dtype=np.float)


def remove_padding(im):
    th = 5000
    diff_pix = np.diff(np.sum(im, axis=0)[::-1])
    idx = np.where(diff_pix > th)[0][0]
    im = np.fliplr(im)
    im[:, 1:-idx + 1] = im[:, idx:]
    return im


fixed = remove_padding(fixed)
moving = remove_padding(moving)

sx, sy, vx, vy = demons(fixed, moving, stop_criterion=1e-3)
warped = iminterpolate(moving, sx=sx, sy=sy)

plt.figure()
plt.title('Fixed')
plt.imshow(fixed)
plt.show()

plt.figure()
plt.title('Moving')
plt.imshow(moving)
plt.show()

plt.figure()
plt.title('Warped')
plt.imshow(warped)
plt.show()
