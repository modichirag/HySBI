import numpy as np
import sys, os
from nbodykit.lab import FFTPower, ArrayMesh

dkfac = 2
BoxSize  = 1000.

def get_dk_power(f):
    BoxSize = 1000
    mesh = ArrayMesh(f/f.mean() - 1, BoxSize=BoxSize)
    kmin2, kF2 = 2*np.pi/BoxSize *dkfac, 2*np.pi/BoxSize *dkfac,     
    pk = FFTPower(mesh, mode='1d', kmin=kmin2, dk=kF2).power
    return pk['k'], pk['power'].real



i0 = int(sys.argv[1])
i1 = int(sys.argv[2])
for i in range(i0, i1):
    print(i)
    f = np.load(f'/mnt/ceph/users/cmodi/Quijote/latin_hypercube_HR/matter/N0256/{i:04d}/field.npy')
    k, p = get_dk_power(f.copy())
    np.save(f'/mnt/ceph/users/cmodi/Quijote/latin_hypercube_HR/matter/N0256/{i:04d}/power-dk{dkfac}.npy', np.stack([k, p], axis=1))
