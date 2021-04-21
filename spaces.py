import numpy as np
import matplotlib.pyplot as plt


class Space:

    def __init__(self, geometry, **kwargs):
        self.dim = 0
        self.size = ()
        if not isinstance(geometry, list):
            self.geometry = [geometry]
        else:
            self.geometry = geometry
        self.prepare_geometry()
        self.extent = self._get_extent()
        self.kwargs = kwargs

    def prepare_geometry(self):
        geometry = self.geometry
        for dimension in geometry:
            self._add_dimension(dimension=dimension)
        pass

    def _add_dimension(self, dimension):
        self.dim += 1
        n = dimension[-1]
        if len(dimension) <= 2:
            xvalues = np.arange(n)
        else:
            xvalues = np.linspace(*dimension)
        self.__setattr__('n' + str(self.dim), n)
        self.__setattr__('xV' + str(self.dim), xvalues)
        self.__setattr__('dx' + str(self.dim), xvalues[1] - xvalues[0])
        self.size += (n, )
        pass

    def _get_volume_element(self):
        V = 1
        for i in range(self.dim):
            V *= self.__getattribute__('dx' + str(i + 1))
        return V

    def get_grid(self):
        mesh = []
        for i in range(self.dim):
            mesh.append(self.__getattribute__('xV' + str(i + 1)))
        return np.stack(np.meshgrid(*mesh))

    def _get_extent(self):
        extent = []
        for i in range(self.dim):
            extent.append(self.geometry[i][0])
            extent.append(self.geometry[i][1])
        return extent

    def frequency_space(self):
        geometry = []
        for i in range(self.dim):
            n = self.__getattribute__("n" + str(i + 1))
            xx = self.__getattribute__("xV" + str(i + 1))
            radius = (xx[-1] - xx[0]) / 2
            max_freq = (n / 2 - 1) * (np.pi / radius)
            min_freq = (-n / 2) * (np.pi / radius)
            geometry.append((min_freq, max_freq, n))
        return Space(geometry=geometry)

    def plot_element(self, f, cmap='bone'):
        plt.figure(figsize=(15, 10))
        if self.dim == 1:
            plt.plot(f)
            plt.xlim(self.extent)
        elif self.dim == 2:
            plt.imshow(np.flipud(f), cmap=cmap, extent=self.extent)
        else:
            raise Exception("Plotting for more than 2 dimensions is not available.")


class LpSpace(Space):
    def __init__(self, geometry, p=2., **kwargs):
        super(LpSpace, self).__init__(geometry, **kwargs)
        self.p = p
        self.geometry = geometry
        self._weight = kwargs.get("weight", None)
        self._volume_element = self._get_volume_element()
        self._precision = kwargs.get("precision", "float32")

    def norm(self, f):
        f = self._cast(f)
        power = np.abs(f) ** self.p
        if self._weight is not None:
            power *= self._weight
        integrate = np.sum(power) * self._volume_element
        return integrate ** (1. / self.p)

    def inner(self, f, g):
        f, g = self._cast(f), self._cast(g)
        corr = f * np.conjugate(g) * self._weight
        return np.sum(corr) * self._volume_element

    def _cast(self, f):
        return f.astype(self._precision)

    def dual(self):
        p = self.p / (self.p - 1)
        return LpSpace(geometry=self.geometry, p=p)

    def frequency_space(self):
        geometry = []
        for i in range(self.dim):
            n = self.__getattribute__("n" + str(i + 1))
            xx = self.__getattribute__("xV" + str(i + 1))
            radius = (xx[-1] - xx[0]) / 2
            max_freq = (n / 2 - 1) * (np.pi / radius)
            min_freq = (-n / 2) * (np.pi / radius)
            geometry.append((min_freq, max_freq, n))
        return LpSpace(geometry=geometry, precision='complex64')


l2 = LpSpace(geometry=(-1, 1, 128))


class SobolevSpace(LpSpace):
    '''Tests'''

    def __init__(self, geometry, s=1, **kwargs):
        super(SobolevSpace, self).__init__(geometry=geometry, p=2, **kwargs)
        self.s = s

    def norm(self, f):
        return 0


h1 = SobolevSpace(geometry=(-1, 1, 128))
h1.norm(np.ones(128))


class Element:
    def __init__(self, space):
        self.space = space
        self.geometry = space.geometry
