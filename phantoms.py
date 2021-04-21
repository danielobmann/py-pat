import numpy as np
from scipy.ndimage import gaussian_filter
import torch


class Phantom2D:
    def __init__(self, space, device='cuda', **kwargs):
        self.space = space
        self.device = device
        self.kwargs = kwargs
        self.grid = space.get_grid()

    @staticmethod
    def circular_mask(grid, r_inner, r_outer):
        mask = np.where(np.sum(grid ** 2, axis=0) > r_inner ** 2, 1, 0)
        mask *= np.where(np.sum(grid ** 2, axis=0) <= r_outer ** 2, 1, 0)
        return mask

    def line_star(self, n, b, r_inner, r_outer, midpoint=np.array([0, 0])):
        gamma = np.pi / n
        grid = (self.grid - midpoint[..., None, None])
        phantom = np.zeros(self.space.size)
        for k in range(n):
            theta = np.array([np.sin(k * gamma), -np.cos(k * gamma)])
            d = np.tensordot(grid, theta, axes=([0], [0]))
            phantom += np.where(np.abs(d) <= b / 2, 1, 0)

        phantom *= self.circular_mask(grid=grid, r_inner=r_inner, r_outer=r_outer)
        phantom = np.clip(phantom, 0, 1)
        return phantom

    def siemens_star(self, n, r_inner, r_outer, midpoint=np.array([0, 0])):
        grid = (self.grid - midpoint[..., None, None])
        PHI = np.arctan(grid[1, ...] / grid[0, ...])
        phi = np.pi / n
        star = (PHI // phi) % 2
        star *= self.circular_mask(grid=grid, r_inner=r_inner, r_outer=r_outer)
        return star

    def delta_peak(self, midpoint=np.array([0, 0]), tol=1e-3):
        grid = self.grid - midpoint[..., None, None]
        return np.where(np.sum(grid ** 2, axis=0) <= tol, 1., 0.)

    def circles(self, n, r_inner, r_outer):
        phi = 2 * np.pi / n
        new_phantom = np.zeros(self.space.size)
        for k in range(n):
            midpoint = np.array([np.cos(k * phi), np.sin(k * phi)]) * r_outer
            dist = np.sqrt(np.sum((self.grid - midpoint[..., None, None]) ** 2, axis=0))
            new_phantom += np.where(dist <= r_inner / 2 * (k + 1) / n, 1, 0)
        return new_phantom

    def newton_rings_(self, r, n):
        grid = self.space.get_grid()
        dist = np.sqrt(np.sum(grid ** 2, axis=0))
        r_inner = r/n
        r_outer = r/(n-1)
        x = np.zeros_like(dist)
        for i in range(n, 1, -2):
            x += np.where((dist <= r_outer) * (dist >= r_inner), 1., 0.)
            r_inner = r/i
            r_outer = r/(i-1)
        return np.clip(x, 0, 1)

    def newton_rings(self, r, base_freq=2, tau=0.5):
        grid = self.space.get_grid()
        dist = np.sqrt(np.sum(grid ** 2, axis=0))
        phantom = np.sin((dist / r * 2 * np.pi * base_freq) ** 2)
        phantom = np.where(phantom >= tau, 1., 0.) * np.where(dist <= r, 1., 0.)
        return phantom

    @staticmethod
    def smooth_phantom(phantom, sigma=1.5):
        return gaussian_filter(phantom, sigma=sigma)

    def to_torch(self, phantom):
        return torch.from_numpy(phantom.astype("float32")[None, None, ...]).to(self.device)
