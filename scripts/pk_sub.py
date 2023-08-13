import numpy as np
import sys, os
from nbodykit.lab import FFTPower, ArrayMesh

def get_pksubs(f, fac=2):

    nc = f.shape[0] // fac 
    psubs = []
    for i in range(fac):
        x0, x1 = nc*i, nc*(i+1)

        for j in range(fac):
            y0, y1 = nc*j, nc*(j+1)

            for ik in range(fac):
                z0, z1 = nc*ik, nc*(ik+1)            
                sub = f[x0:x1, y0:y1, z0:z1]
                box = sub*1.
                mesh = ArrayMesh(box/box.mean(), BoxSize=1000/fac)
                pk = FFTPower(mesh, mode='1d', dk=2*np.pi/1000*fac, kmin=2*np.pi/1000*fac ).power
                psubs.append(pk['power'])
    psubs = np.array(psubs)
    return pk['k'], psubs


fac = 2
i0 = int(sys.argv[1])
i1 = int(sys.argv[2])
for i in range(i0, i1):
    print(i)
    f = np.load(f'/mnt/ceph/users/cmodi/Quijote/latin_hypercube_HR/matter/N0256/{i:04d}/field.npy')
    ksubs, psubs = get_pksubs(f, fac=fac)
    np.save(f'/mnt/ceph/users/cmodi/Quijote/latin_hypercube_HR/matter/N0256/{i:04d}/power_split{fac}-dk2.npy', psubs)
    np.save(f'/mnt/ceph/users/cmodi/Quijote/latin_hypercube_HR/matter/N0256//k_split{fac}-dk2.npy', ksubs)
