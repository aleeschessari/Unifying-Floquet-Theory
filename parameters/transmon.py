from qutip import *
import scqubits
import numpy as np

dim = 110

EC = 0.264/31.3
flux = 0

Ejeff = np.abs(np.cos(np.pi*flux))

transmon = scqubits.TunableTransmon(
               EJmax=1,
               EC=EC,
               d=0,
               flux=flux,
               ng=0.0,
               ncut=dim,
               truncated_dim=dim
)

H_sys = Qobj(transmon.hamiltonian(energy_esys=True))

drive_op = Qobj(transmon.n_operator(energy_esys=True))

wq = H_sys.eigenenergies()[1]-H_sys.eigenenergies()[0]

num_A = 70

g = 0.130/31.3/1.4
kappa = 0.002/31.3

fname = '../data/params/transmon.npz'
np.savez(fname, drive_op=drive_op.full(), wq=wq, H_sys=H_sys.full(),\
    dim=dim, num_A=num_A, g=g, kappa=kappa, Ejeff=Ejeff, EC=EC, flux=flux)

# CONVERGENZA 25 STATI, 20 REPLICHE
# 25 x 25 per spettri