import numpy as np
import sys, os
import galactic_wavelets as gw
import torch

i0, i1 = int(sys.argv[1]), int(sys.argv[2])

device = 'cpu'
BoxSize = 1000
N = 256
df_shape = (N, N, N)

dx = BoxSize/N
kmax = np.pi/dx
kF = 2*np.pi/BoxSize
kN = N *kF / 2
print("kF, kmax, kN : ", kF, kmax, kN)

J = 7
Q = 2
moments = [1/2, 1, 1.5, 2]
M = len(moments)
kcpifac = 1.
kc = kcpifac * np.pi # Cutoff frequency of the mother wavelet (in units of 2 px^-1)
erosion_threshold = 0.1
fname = f"M{M}_J{J}_Q{Q}_e{erosion_threshold}_kc{kcpifac}"
save_path = f'/mnt/ceph/users/cmodi/Quijote/latin_hypercube_HR/matter/N0256/'
print(f"File name : {fname}")


def save_wavelets(wst_op_full):
    if os.path.isfile(f"{save_path}/wx_renorm_{fname}.npz"):
        print("Wavelets already saved")        
        return None
    
    fw = torch.fft.fftshift(wst_op_full.wt_op.get_wavelets("fourier").squeeze(), dim=(-3, -2, -1))
    w = torch.fft.fftshift(wst_op_full.wt_op.get_wavelets("physical").squeeze(), dim=(-3, -2, -1))

    w_renorm = w.clone()
    for j in range(J):
        for q in range(Q):
            w_renorm[j*Q + q] *= 2**(3*(j*Q + q)/Q)
    aw_renorm = torch.absolute(w_renorm)
    w_max = torch.amax(aw_renorm, dim=(-3, -2, -1), keepdim=True)
    w_supports = aw_renorm > erosion_threshold*w_max

    sampling_step = np.pi/kmax
    xvals = np.arange(-N//2, N//2)*sampling_step
    kvals = np.fft.fftshift(2*np.pi*np.fft.fftfreq(N, sampling_step))
    
    xdict, xlimdict, fdict = {}, {}, {}
    for j in range(J):
        for q in range(Q):
            support = torch.from_numpy(np.argwhere(w_supports[j*Q + q, N//2, N//2, :].numpy()))
            support_min = -(N/2 - torch.min(support[support[:, 0] < N/2]))
            support_max = torch.max(support[support[:, 0] >= N/2]) - N/2
            key = f"$j={j} + {q}/{Q}$" if q != 0 else f"$j={j}$"
            # print(xvals.shape, w_renorm[j*Q + q, N//2, N//2, :].cpu().flatten().shape)
            xdict[key] = np.array([xvals, w_renorm[j*Q + q, N//2, N//2, :].numpy()])
            fdict[key] = np.array([kvals, fw[j*Q + q, N//2, N//2, :].numpy()])
            xlimdict[key] = [support_min.cpu()*sampling_step, support_max.cpu()*sampling_step]

    np.savez(f"{save_path}/wx_renorm_{fname}.npz", **xdict)
    np.savez(f"{save_path}/wf_renorm_{fname}.npz", **fdict)
    np.savez(f"{save_path}/wsupports_{fname}.npz", **xlimdict)


print("Setup scattering op for full box")
wst_op_full = gw.ScatteringOp(df_shape, J, Q,
                              moments=moments,
                              kc=kc,
                              device=device)
print("Setup scattering op for half box")
wst_op_half = gw.ScatteringOp(tuple([x//2 for x in df_shape]), J, Q,
                              moments=moments,
                              kc=kc,
                              erosion_threshold=erosion_threshold,
                              device=device)
save_wavelets(wst_op_full)


for lh in range(i0, i1):
    mesh = torch.from_numpy(np.load('/mnt/ceph/users/cmodi/Quijote/latin_hypercube_HR/matter/N0256/%04d/field.npy'%lh))
    mesh /= mesh.mean()

    print("do for full")
    s0_full, s1_full = wst_op_full(mesh)
    s0_full = s0_full.reshape(1, -1)
    s1_full = s1_full.reshape(1, M, J, Q)
    print("do for halfs")
    s0_halfs, s1_halfs = [], []
    for i in range(8):
        k, l, m = i // 4, (i % 4) // 2, i % 2
        df_half = mesh[k*N//2:(k+1)*N//2, l*N//2:(l+1)*N//2, m*N//2:(m+1)*N//2]
        s0_half, s1_half = wst_op_half(df_half)
        s0_halfs.append(s0_half)
        s1_halfs.append(s1_half)
    s0_halfs, s1_halfs = torch.stack(s0_halfs), torch.stack(s1_halfs)
    s1_halfs = s1_halfs.reshape(8, M, J, Q)

    wavs = {'s0_full':s0_full, 's1_full':s1_full, 's0_halfs':s0_halfs, 's1_halfs':s1_halfs}
    np.savez(f"{save_path}/{lh:04d}/{fname}.npz", **wavs)
