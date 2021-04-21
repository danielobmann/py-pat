from pypat.spaces import LpSpace
from pypat.wave_operator import WaveOperator, FilteredBackprojection
from pypat.phantoms import Phantom2D

import torch
import matplotlib.pyplot as plt
from geometry import *

dev = torch.device("cuda")
Nx_sim = 384
geometry = 2 * [(-2*REGION, 2*REGION, Nx_sim)]
space = LpSpace(geometry=geometry)

measurements_points = np.stack([RADIUS * np.cos(THETA), RADIUS * np.sin(THETA)])
A = WaveOperator(space=space, measurement_points=measurements_points, Tmax=Tmax, device=dev)
FBP = FilteredBackprojection(wave_operator=A, device=dev)
Phantom = Phantom2D(space=space)

b = REGION / 10
r_inner = REGION / 8
r_outer = REGION / 2

test = Phantom.siemens_star(2, r_inner=r_inner, r_outer=r_outer)
test = Phantom.smooth_phantom(test)
plt.imshow(test)

x = Phantom.to_torch(test)
y = A(x)
xfbp = FBP(y[0, 0, ...].cpu().numpy())
g = A.adjoint(y)

plt.subplot(131)
plt.imshow(x.cpu()[0, 0, ...])

plt.subplot(132)
plt.imshow(y.cpu()[0, 0, ...], aspect='auto')

plt.subplot(133)
plt.imshow(g.cpu()[0, 0, ...])
