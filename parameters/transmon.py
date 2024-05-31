from qutip import *
import scqubits
import numpy as np

dim = 12

fluxonium = scqubits.TunableTransmon(
               EJmax=1,
               EC=0.264/34,
               d=0,
               flux=0.127,
               ng=0.0,
               truncated_dim=10,
               ncut=dim
)

n_states = 4

H_sys = Qobj(fluxonium.hamiltonian(energy_esys=True)[0:n_states,0:n_states])

drive_op = Qobj(fluxonium.n_operator(energy_esys=True)[0:n_states,0:n_states])

wq = H_sys.eigenenergies()[1]-H_sys.eigenenergies()[0]

N_rep = 10
N_fock = 30

g = 0.130/34
kappa = 0.002/34