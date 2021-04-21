import numpy as np
import torch
import torch.fft as fft
from tqdm import tqdm
import matplotlib.pyplot as plt
from geometry import *
from pypat.spaces import Space

# TODO: MeasurementGeometry --> ???
# TODO: Circle geometry (subclass) --> evaluation (either pointwise or interpolation), normal of geometry
# TODO: Change wave operator to use evaluation of a MeasurementGeometry


class Measurements:
    def __init__(self, space, measurement_points, Tmax, **kwargs):
        assert measurement_points.shape[0] == space.dim

        self.space = space
        self.kwargs = kwargs
        self.measurement_poins = measurement_points
        self.Nphi = measurement_points.shape[-1]
        self.indices = self._find_indices()
        self.Tmax = Tmax
        self.c0 = kwargs.get("c0", 1491.2)
        self.Nt = np.ceil(1 + Tmax * self.c0 * (self.space.n1 - 1) / (2 * self.space.xV1[-1])).astype(int)
        self.time = np.linspace(0, Tmax, self.Nt)
        self.dt = self.time[1] - self.time[0]

    def _find_indices(self):
        '''
        Calculate location of nearest neighbors of measurement points in given space.
        :return: List of indices where the measurement points are located
        '''
        grid = self.space.get_grid()
        indices = []

        for i in range(self.Nphi):
            point = self.measurement_poins[:, i]
            distance = lambda x, y: np.sqrt((x - point[0]) ** 2 + (y - point[1]) ** 2)
            dist = distance(*grid)
            indices.append(np.unravel_index(np.argmin(dist, axis=None), dist.shape))
        indices = tuple(np.array(list(zip(*indices))))
        return indices

    def plot_geometry(self):
        assert self.space.dim == 2
        x = np.zeros(self.space.size)
        x[self.indices] = 1
        self.space.plot_element(x)
        pass


class WaveOperator(torch.nn.Module):
    # TODO: Perfectly matched layer for faster computation?
    # TODO: Use interpolation to allow more angular samples
    '''
    https://arxiv.org/pdf/1611.07563.pdf
    '''

    def __init__(self, space, measurement_points, Tmax, device='cuda', **kwargs):
        super(WaveOperator, self).__init__()
        self.space = space
        self.space_outer = Space(geometry=[tuple([2 * d for d in dim]) for dim in space.geometry])
        self.measurements = Measurements(space=self.space_outer, measurement_points=measurement_points, Tmax=Tmax,
                                         **kwargs)
        self.device = device
        self.kwargs = kwargs
        self.sin_filter, self.sinc_filter = self._compute_filters()
        self._pad_size = self._get_padding_size()
        self._pad = torch.nn.ConstantPad2d(self._pad_size, 0)
        self.fourier_dim = (-2, -1)

    def forward(self, f):
        initial = self._pad(f)
        return self._solve_forward(initial_pressure=initial)

    def adjoint(self, g):
        f = self._solve_backward(source=g)
        return (f[..., 1] - f[..., 0]) / self.measurements.dt

    def _compute_filters(self):
        kspace = self.space_outer.frequency_space()
        kgrid = kspace.get_grid()
        k_norm = np.sqrt(np.sum(kgrid ** 2, axis=0))
        k_norm *= self.measurements.c0 * self.measurements.dt / 2

        sin_fil = np.sin(k_norm) ** 2
        sin_fil = np.fft.ifftshift(sin_fil).astype("float32")
        sin_fil = torch.from_numpy(sin_fil[None, None, ...]).to(self.device)

        sinc_fil = np.sinc(k_norm) ** 2 * (self.measurements.c0 * self.measurements.dt / 2) ** 2
        sinc_fil = np.fft.ifftshift(sinc_fil).astype("float32")
        sinc_fil = torch.from_numpy(sinc_fil[None, None, ...]).to(self.device)
        return sin_fil, sinc_fil

    def _solve_forward(self, initial_pressure):
        b, c = initial_pressure.shape[0], initial_pressure.shape[1]
        W = initial_pressure
        Walt = W.clone()
        y = torch.zeros(b, c, self.measurements.Nphi, self.measurements.Nt).to(self.device)
        for i in range(self.measurements.Nt):
            FW = fft.ifftn(fft.fftn(W, dim=self.fourier_dim) * self.sin_filter, dim=self.fourier_dim)
            Wneu = 2 * W - Walt - 4 * FW
            Walt = W
            W = Wneu
            y[:, :, :, i] = W[:, :, self.measurements.indices[0], self.measurements.indices[1]].real
        return y

    def _get_current_source(self, time_samples):
        b, c = time_samples.shape[0], time_samples.shape[1]
        s = torch.zeros(b, c, *self.space_outer.size).to(self.device)
        s[:, :, self.measurements.indices[0], self.measurements.indices[1]] = time_samples
        return s

    def _solve_backward(self, source):
        b, c = source.shape[0], source.shape[1]
        WW = torch.zeros(b, c, *self.space.size, 2).to(self.device)
        W = torch.zeros(b, c, *self.space_outer.size).to(self.device)
        Walt = W.clone()
        for i in range(self.measurements.Nt - 1, -1, -1):
            s = self._get_current_source(-source[:, :, :, i] / self.space_outer.dx1)
            FS = fft.ifftn(fft.fftn(s, dim=self.fourier_dim) * self.sinc_filter, dim=self.fourier_dim)
            FW = fft.ifftn(fft.fftn(W, dim=self.fourier_dim) * self.sin_filter, dim=self.fourier_dim)
            Wneu = 2 * W - Walt - 4 * (FW - FS)
            Walt = W
            W = Wneu
            if i <= 1:
                size1lower = self._pad_size[0]
                size1upper = self._pad_size[0] + self.space.geometry[0][-1]

                size2lower = self._pad_size[1]
                size2upper = self._pad_size[1] + self.space.geometry[1][-1]

                WW[..., i] = W[:, :, size1lower:size1upper, size2lower:size2upper].real
        return WW

    def _get_padding_size(self):
        padding = []
        for i in range(self.space.dim):
            Nx = self.space.__getattribute__('n' + str(i + 1))
            Nx_outer = self.space_outer.__getattribute__('n' + str(i + 1))
            padding.append((Nx_outer - Nx) // 2)
            padding.append((Nx_outer - Nx) // 2)
        return tuple(padding)


class FilteredBackprojection:
    # TODO: Implement in pytorch and gpu
    # TODO: Find error with negative sign
    def __init__(self, wave_operator, device='cuda', **kwargs):
        self.wave_operator = wave_operator
        self.measurements = wave_operator.measurements
        self.device = device
        self.kwargs = kwargs

        self.Nt = self.measurements.Nt
        self.Ndet = self.measurements.Nphi
        self.Nt1 = self.Nt - 1
        self.time = self.measurements.time

        self.AA, self.BB = self._prepare_filters()

    def _prepare_filters(self):
        t = self.measurements.time

        AA = np.zeros((self.Nt1, self.Nt1))
        BB = np.zeros((self.Nt1, self.Nt1))

        for m in range(self.Nt1):
            for n in range(m, self.Nt1):
                AA[m, n] = np.log(t[n + 1] + np.sqrt(t[n + 1] ** 2 - t[m] ** 2))
                AA[m, n] -= np.log(t[n] + np.sqrt(t[n] ** 2 - t[m] ** 2))
                BB[m, n] = -t[n] * AA[m, n] + np.sqrt(t[n + 1] ** 2 - t[m] ** 2) - np.sqrt(t[n] ** 2 - t[m] ** 2)

        return AA, BB

    def _prepare_data(self, g):
        t = self.time
        dt = t[1] - t[0]
        c0 = self.wave_operator.measurements.c0
        qq = np.zeros((self.Nt1, self.Ndet))
        qqh = np.zeros((self.Nt1, self.Ndet))

        for k in range(self.Ndet):
            for m in range(self.Nt1):
                if m == 0:
                    qq[m, k] = 0
                elif m == 1:
                    qq[m, k] = (g[k, m + 1] / t[m + 1] - g[k, m] / t[m]) / dt / c0 ** 2
                else:
                    qq[m, k] = (g[k, m + 1] / t[m + 1] - g[k, m - 1] / t[m - 1]) / 2 / dt / c0 ** 2

        Nt1m = self.Nt1 - 1
        for k in range(self.Ndet):
            qqh[:, k] = self.AA[:, 0:Nt1m] @ qq[0:Nt1m, k]
            qqh[:, k] += (self.BB[:, 0:Nt1m] @ qq[1:self.Nt1, k] - self.BB[:, 0:Nt1m] @ qq[0:Nt1m, k]) / dt
        return qqh

    def __call__(self, g, *args, **kwargs):
        size = self.wave_operator.space.size
        p0 = np.zeros(size)
        grid = self.wave_operator.space.get_grid()
        Nt1m = self.Nt1 - 1

        qqh = self._prepare_data(g)
        t = self.time
        c0 = self.wave_operator.measurements.c0
        dt = t[1] - t[0]
        points = self.measurements.measurement_poins

        for i1 in range(size[0]):
            for j1 in range(size[1]):
                for k1 in range(self.Ndet):
                    det_x, det_y = points[0, k1], points[1, k1]
                    x, y = grid[0, i1, j1], grid[1, i1, j1]
                    rho_grid = np.sqrt((det_x - x) ** 2 + (det_y - y) ** 2)
                    cos_gamma = det_x * (det_x - x) + det_y * (det_y - y)
                    mm = int(np.minimum(np.floor(rho_grid / c0 / dt) + 1, Nt1m - 1))
                    q = qqh[mm, k1] + (rho_grid / c0 - t[mm]) * (qqh[mm + 1, k1] - qqh[mm, k1]) / dt
                    p0[i1, j1] = p0[i1, j1] + q * cos_gamma * 2 * np.pi / self.Ndet
        return p0
