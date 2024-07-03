from qutip import *
import scqubits
import numpy as np

dim = 110
n_states = 20

EC = 1/4
EL = 1/4
flux = 0.5

fluxonium = scqubits.Fluxonium(EJ = 1,
                           EC = EC,
                           EL = EL,
                           flux = flux,
                           cutoff = dim,
                           truncated_dim = dim)

H_sys = Qobj(fluxonium.hamiltonian(energy_esys=True))

drive_op = Qobj(fluxonium.n_operator(energy_esys=True))

wq = H_sys.eigenenergies()[1]-H_sys.eigenenergies()[0]

num_A = 80

g  = 0.02/4
kappa = 0.001/4

fname = 'data/params/fluxonium.npz'
np.savez(fname, drive_op=drive_op.full(), wq=wq, H_sys=H_sys.full(),\
    dim=dim, num_A=num_A, g=g, kappa=kappa, EC=EC, EL=EL, flux=flux)

# convergenza 10 stati 5 repliche