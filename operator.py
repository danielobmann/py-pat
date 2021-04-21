import numpy as np


class Operator:
    def __init__(self, domain, codomain=None):
        self.domain = domain
        if codomain is None:
            self.codomain = domain
        else:
            self.codomain = codomain

    def get_domain(self):
        return self.domain

    def get_codomain(self):
        return self.codomain


class FourierTransform(Operator):
    def __init__(self, domain, codomain):
        super(FourierTransform, self).__init__(domain=domain, codomain=codomain)

    def __call__(self, f):
        return np.fft.fftn(f, norm='ortho')

    def adjoint(self, g):
        return np.fft.ifftn(g, norm='ortho')


l2 = LpSpace((-1, 1, 512))
f = lambda t: np.exp(-t ** 2)
fourier = FourierTransform(domain=l2, codomain=l2.frequency_space())
fv = f(l2.xV1)
g = fourier(fv)
finv = fourier.adjoint(g)

freq = np.fft.fftshift(fourier.codomain.xV1)
fprime = fourier.adjoint((freq / 2) ** 2 * g)
plt.plot(fprime)


def test_norm():
    f = np.random.normal(0, 1, 512)
    g = fourier(f)
    return fourier.codomain.norm(g) / fourier.domain.norm(f)

# TODO: Create composition of operators
# TODO: Create Compressed Sensing operator
# TODO: Copy wave operator here and implement with spaces
# TODO: Implement everything using torch
