import numpy as np
from nbodykit.lab import FFTPower, ArrayMesh
from nbodykit.mockmaker import gaussian_real_fields
import sys, os

seed = int(sys.argv[1])

grid              = 256    #grid size
BoxSize           = 1000.0 #Mpc/h

# Define fundamental and Nyquist modes
kF =  2*np.pi/BoxSize
dk, kmin = kF, kF
kN = grid * kF

fac = 2
save_path = f"/mnt/home/cmodi/Research/Projects/HySBI/data/conv_matrix_f{fac}/"
os.makedirs(save_path, exist_ok=True)
mesh = ArrayMesh(np.zeros([grid, grid, grid]).astype(np.float32)*0., BoxSize)
msize = FFTPower(mesh, mode='1d', kmin=kmin*fac, dk=kF*fac).power['k'].size
print("Expected shape of matrix : ", msize)

## Compute mask
x_arr = np.arange(grid)
filt1d = np.empty(grid)
filt1d[x_arr >= grid//2] = 0.
filt1d[x_arr< grid//2] = 1.
filt = np.asarray(np.meshgrid(*[filt1d for _ in range(3)])).prod(axis=0)
assert filt.sum() == grid**3/8

def compute_pk_out(pk_index, seed, kmax=kN, fac=2):

    kmin, kF = 2*np.pi/BoxSize *fac, 2*np.pi/BoxSize *fac, 
    k1, k2 = kmin + kF*(pk_index), kmin + kF*(pk_index + 1) 
    klow, khigh = k1 - 1e-4, k2 - 1e-4

    if k2> kmax:
        print(f"Upper edge of the bin = {k2}, is greater than kmax = {kmax}")
        raise 
    if klow >= kmin:        
        k_arr = list(np.arange(kmin, klow, kF)) + [klow, k1, khigh] +list(np.arange(k2, kmax, kF))
    else:    
        print('klow < kmin')
        k_arr = [klow, k1, khigh] +list(np.arange(k2, kmax, kF))
    
    k_arr = np.asarray(k_arr)
    pk_arr = np.zeros_like(k_arr)
    pk_arr[(k_arr>klow) & (k_arr<k2)] = 1.    

    mesh = ArrayMesh(np.zeros([grid, grid, grid]).astype(np.float32)*0., BoxSize)
    ipk = lambda x: np.interp(x, k_arr, pk_arr)
    density_r = gaussian_real_fields(mesh.pm, ipk, seed + pk_index*123, unitary_amplitude=True)[0]       
    density_r = density_r[...]
    
    # Now cut this down and zero-pad it , remove mean
    density_cut = density_r*filt
    density_cut -= np.mean(density_cut)
 
    Pk = FFTPower(ArrayMesh(density_cut.astype(np.float32), BoxSize), mode='1d', kmin=kmin, dk=kF).power
    k_out = Pk['k']
    Pk_out = Pk['power'].real
    Nmodes = Pk['modes']
    
    return k_arr, pk_arr, k_out[k_out<kmax], Pk_out[k_out<kmax]


Pk_mat = np.zeros((msize, msize))
print("Using seed %d"%seed)
for index in range(msize):
    if index % 10 == 0: print(f"for index : {index}")
    Pk_mat[index] = compute_pk_out(index, seed)[-1]
np.save(f'{save_path}/M{seed}', Pk_mat)

