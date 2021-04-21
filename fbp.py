import numpy as np
import matplotlib.pyplot as plt

x_mid = 0e-3
y_mid = 0e-3
x_regionSize = 10.0e-3
y_regionSize = 10.0e-3
recxMin = x_mid - x_regionSize
recxMax = x_mid + x_regionSize
recyMin = y_mid - y_regionSize
recyMax = y_mid + y_regionSize

NRx = 512
NRy = 512 # round(NRx*(recyMax - recyMin)/(recxMax - recxMin))

c00 = 1491.2
# -----


def read_ifs_data(path, sensors=64):
    row_count = sum(1 for _ in open(path, "r")) - 2
    y = np.zeros((sensors+1, row_count), np.float32)

    with open(path, "r") as file:
        _ = file.readline()
        active = [int(s) for s in file.readline().split(',')]
        for i, line in enumerate(file):
            lstrip = line.rstrip('\n')
            temp = np.array([float(val) for val in lstrip.split(',')])
            y[active, i] = temp
    return y, [j - 1 for j in active if j != 0]


fn = "data/real/all/Gitter1mm_150QSD_10mJ_00_00.ifs"
# fn = "data/real/all/sphere_P7_AVG64_2.ifs"
# fn = "data/real/all/NetzAgarose_700.ifs"
fn = "data/real/all/Blatt_3_AVG64_1.ifs"

datx16, activeFibers = read_ifs_data(fn)

# -----
ind_off = 100
ind_end = len(datx16[0, :])-1
fadeLen = 0

t_data_start = datx16[0, ind_off - fadeLen]
t_data_end = datx16[0, ind_end + fadeLen]
t_end = 50e-6
if t_end < t_data_end:
    t_end = t_data_end

dt = (datx16[0, -1] - datx16[0, 0])/(len(datx16[0, ]) - 1)

# TODO: removeBG
removeBG = 0
if removeBG:
    pass
else:
    data0 = datx16[1:, ind_off - fadeLen: ind_end + fadeLen]

for i in range(len(activeFibers)):
    data0[i, ] = data0[i, ] - np.mean(data0[i, ])
    # data0[i, ] = data0[i, ]/np.std(data0[i, ])

# TODO: filtit
filtit = 0
if filtit:
    pass

# TODO: norreal
norreal = 0
if norreal:
    pass

sensor_radius = 40e-3
sensor_angle = 4.5861*63/180*np.pi
num_sensor_points = 253
r = np.ones((1, num_sensor_points)) * sensor_radius
phi = np.pi/2 + 0.6202525 + np.linspace(0, 4.5861 * 63/180*np.pi, num_sensor_points)


# TODO: pol2cart
def pol2cart(phi, rho):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


det_x, det_y = pol2cart(phi[::4], r[0, ::4])
det_x = det_x[activeFibers]
det_y = det_y[activeFibers]

# TODO: doFade
doFade = 0
if doFade:
    pass

tt = -14
t = np.arange(0, t_end + dt*tt, step=dt)
tLen = len(t)

t_offset = tt*dt
data = np.concatenate([np.zeros((data0.shape[0], round(t_data_start/dt) + tt)), data0], axis=1)
dataLen = len(data[0, ])

if tLen > dataLen:
    data = np.concatenate([data, np.zeros((data.shape[0], tLen - dataLen))], axis=1)

data = data[activeFibers, :]
# TODO: Figure
plt.imshow(data, aspect='auto', cmap="bone")
plt.xlabel("Time [sec]")
plt.ylabel("Sensor")
plt.colorbar()
plt.savefig("images/Blatt_3_AVG64_1_data.pdf", dpi=500)


# Precompute (--> Make this more efficient!)
Nt = data.shape[1]
Ndet = data.shape[0]

Nt1 = Nt - 1
AA = np.zeros((Nt1, Nt1))
BB = np.zeros((Nt1, Nt1))

for m in range(ind_off, Nt1):
    for n in range(m, Nt1):
        AA[m, n] = np.log(t[n+1] + np.sqrt(t[n+1]**2 - t[m]**2))
        AA[m, n] -= np.log(t[n] + np.sqrt(t[n]**2 - t[m]**2))
        BB[m, n] = -t[n] * AA[m, n] + np.sqrt(t[n+1]**2 - t[m]**2) - np.sqrt(t[n]**2 - t[m]**2)


# ----------------------
# Smoothing of data

def smoothing(data, eps, dt=dt):
    datahat = np.fft.fft(data)
    freq = np.fft.fftfreq(data.shape[-1], dt)
    chi = np.where(np.abs(freq) <= eps, 1, 0.)
    chi = np.expand_dims(chi, axis=0)
    datasmooth = np.fft.ifft(datahat * chi)
    return np.real(datasmooth)


eps = c00/(1e-3*np.sqrt(8))
data = smoothing(data, eps=eps)


# Filter the data (--> Matrix multiplication?)
dci = 1
c0 = c00
qq = np.zeros((Nt1, Ndet))
qqh = np.zeros((Nt1, Ndet))

for k in range(Ndet):
    for m in range(ind_off, Nt1):
        if m == 0:
            qq[m, k] = 0
        elif m == 1:
            qq[m, k] = (data[k, m+1]/t[m+1] - data[k, m]/t[m])/dt/c0**2
        else:
            qq[m, k] = (data[k, m + 1] / t[m + 1] - data[k, m-1] / t[m-1]) / 2 / dt / c0 ** 2


Nt1m = Nt1 - 1
for k in range(Ndet):
    qqh[:, k] = AA[:, 0:Nt1m] @ qq[0:Nt1m, k] + (BB[:, 0:Nt1m] @ qq[1:Nt1, k] - BB[:, 0:Nt1m] @ qq[0:Nt1m, k])/dt


# Reconstruction (--> Matrix multiplication?)
dxRec = (recxMax - recxMin)/(NRx - 1)
dyRec = (recyMax - recyMin)/(NRy - 1)

rekoGridx = np.arange(recxMin, recxMax+dxRec, dxRec)
rekoGridy = np.arange(recyMin, recyMax+dyRec, dyRec)
p0 = np.zeros((NRy, NRx))

for i1 in range(NRx):
    print(i1)
    for j1 in range(NRy):
        for k1 in range(Ndet):
            rho_grid = np.sqrt((det_x[k1] - rekoGridx[i1])**2 + (det_y[k1] - rekoGridy[j1])**2)
            cos_gamma = det_x[k1]*(det_x[k1] - rekoGridx[i1]) + det_y[k1]*(det_y[k1] - rekoGridy[j1])
            mm = int(np.minimum(np.floor(rho_grid/c0/dt) + 1, Nt1m-1))
            q = qqh[mm, k1] + (rho_grid/c0 - t[mm])*(qqh[mm+1, k1] - qqh[mm, k1])/dt
            p0[j1, i1] = p0[j1, i1] + q*cos_gamma*2*np.pi/Ndet


# Plot reconstruction
m_to_mm = 1000
plt.imshow(p0, cmap='bone', extent=[recxMin*m_to_mm, recxMax*m_to_mm, recyMin*m_to_mm, recyMax*m_to_mm])
plt.colorbar()
plt.xlabel("x [mm]")
plt.ylabel("y [mm]")
plt.savefig("images/Blatt_3_AVG64_1_FBP.pdf", dpi=500)


np.save("data/reconstructions/Blatt_3_AVG64_1_FBP.npy", p0)
