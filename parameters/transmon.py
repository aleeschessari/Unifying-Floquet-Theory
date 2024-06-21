from qutip import *
import scqubits
import numpy as np

dim = 20

EC = 0.264/34
flux = 0.127

Ejeff = np.abs(np.cos(np.pi*flux))

n_states = 4

transmon = scqubits.TunableTransmon(
               EJmax=1,
               EC=EC,
               d=0,
               flux=flux,
               ng=0.0,
               ncut=dim,
               truncated_dim=n_states
)

H_sys = Qobj(transmon.hamiltonian(energy_esys=True))

drive_op = Qobj(transmon.n_operator(energy_esys=True))

wq = H_sys.eigenenergies()[1]-H_sys.eigenenergies()[0]

N_rep = 30 # this means that we will have 2*N_rep+1 replicas
N_fock = 20

num_A = 80

g = 0.130/34/1.4
kappa = 0.002/34