import numpy as np
import sys, os
from nbodykit.lab import FFTPower, ArrayMesh
from scipy.interpolate import InterpolatedUnivariateSpline as iuspline

fac = 2
dkfac = 1
BoxSize  = 1000.

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
                pk = FFTPower(mesh, mode='1d', dk=2*np.pi/1000*dkfac, kmin=2*np.pi/1000*fac ).power
                psubs.append(pk['power'])
    psubs = np.array(psubs)
    return pk['k'], psubs


def get_pksubs_sym(f, fac=2):

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

                box = np.pad(box, (0, box.shape[0]-1), mode='symmetric')            
                print('mean : ', box.mean())
                mesh = ArrayMesh(box/box.mean(), BoxSize=1000/256*box.shape[0])
                pk = FFTPower(mesh, mode='1d', dk=2*np.pi/1000, kmin=2*np.pi/1000*fac ).power
                psubs.append(pk['power'])
                
    psubs = np.array(psubs)
    return pk['k'], psubs


def get_pksubs_deconv(f, fac=2):

    ms = np.stack([np.load(f'../data/conv_matrix_f{fac}/M{i}.npy') for i in range(50)] , axis=0)
    mm = ms.mean(axis=0)
    imm = np.linalg.inv(mm).T

    mesh = ArrayMesh(f / f.mean(), BoxSize=BoxSize)
    kmin, kF = 2*np.pi/BoxSize, 2*np.pi/BoxSize     
    ikvals = FFTPower(mesh, mode='1d', kmin=kmin, dk=kF).power['k']

    nc = f.shape[0] // fac 
    pad = f.shape[0] - nc
    psubs = []
    ipsubs = []
    for i in range(fac):
        x0, x1 = nc*i, nc*(i+1)
        
        for j in range(fac):
            y0, y1 = nc*j, nc*(j+1)
    
            for ik in range(fac):
                z0, z1 = nc*ik, nc*(ik+1)

                print(x0, x1, y0, y1, z0, z1)
                sub = f[x0:x1, y0:y1, z0:z1]
                sub = sub/sub.mean() - 1.
                box = np.pad(sub, ((0, pad), (0, pad), (0, pad)))
                mesh = ArrayMesh(box , BoxSize=BoxSize)

                kmin2, kF2 = 2*np.pi/BoxSize *fac, 2*np.pi/BoxSize *fac,     
                pk = FFTPower(mesh, mode='1d', kmin=kmin2, dk=kF2).power
                psubs.append(imm @ pk['power'].real)
                ksubs = pk['k']
                ipsubs = iuspline(ksubs, psubs[-1], ext=1)(ikvals)
                
    psubs = np.array(psubs)
    ipsubs = np.array(ipsubs)
    
    # interpolate
    return pk['k'], psubs, ikvals, ipsubs


i0 = int(sys.argv[1])
i1 = int(sys.argv[2])
for i in range(i0, i1):
    print(i)
    f = np.load(f'/mnt/ceph/users/cmodi/Quijote/latin_hypercube_HR/matter/N0256/{i:04d}/field.npy')
    #ksubs, psubs = get_pksubs(f, fac=fac)
    #np.save(f'/mnt/ceph/users/cmodi/Quijote/latin_hypercube_HR/matter/N0256/{i:04d}/power_split{fac}-dk{dkfac}.npy', psubs)
    #np.save(f'/mnt/ceph/users/cmodi/Quijote/latin_hypercube_HR/matter/N0256//k_split{fac}-dk{dkfac}.npy', ksubs)

    # ksubs, psubs = get_pksubs_sym(f, fac=fac)
    # np.save(f'/mnt/ceph/users/cmodi/Quijote/latin_hypercube_HR/matter/N0256/{i:04d}/power_split{fac}-sym.npy', psubs)
    # np.save(f'/mnt/ceph/users/cmodi/Quijote/latin_hypercube_HR/matter/N0256//k_split{fac}-sym.npy', ksubs)

    ksubs, psubs, iksubs, ipsubs = get_pksubs_deconv(f, fac=fac)
    np.save(f'/mnt/ceph/users/cmodi/Quijote/latin_hypercube_HR/matter/N0256/{i:04d}/power_split{fac}-deconv.npy', psubs)
    np.save(f'/mnt/ceph/users/cmodi/Quijote/latin_hypercube_HR/matter/N0256/{i:04d}/power_split{fac}-deconv-interp.npy', ipsubs)
    np.save(f'/mnt/ceph/users/cmodi/Quijote/latin_hypercube_HR/matter/N0256//k_split{fac}-deconv.npy', ksubs)
    np.save(f'/mnt/ceph/users/cmodi/Quijote/latin_hypercube_HR/matter/N0256//k_split{fac}-deconv-interp.npy', iksubs)
