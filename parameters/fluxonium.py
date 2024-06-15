from qutip import *
import scqubits
import numpy as np

dim = 110

fluxonium = scqubits.Fluxonium(EJ = 1,
                           EC = 1/4,
                           EL = 1/4,
                           flux = 0.5,
                           cutoff = dim)

n_states = 4

H_sys = Qobj(fluxonium.hamiltonian(energy_esys=True)[0:n_states,0:n_states])

drive_op = Qobj(fluxonium.n_operator(energy_esys=True)[0:n_states,0:n_states])

wq = H_sys.eigenenergies()[1]-H_sys.eigenenergies()[0]

N_rep = 10
N_fock = 30

num_A = 40

g  = 0.02/4
kappa = 0.001/4