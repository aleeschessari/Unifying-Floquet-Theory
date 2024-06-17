from qutip import *
import scqubits
import numpy as np

dim = 11

EC = 0.264/34
flux = 0.127

Ejeff = np.abs(np.cos(np.pi*flux))

transmon = scqubits.TunableTransmon(
               EJmax=1,
               EC=EC,
               d=0,
               flux=flux,
               ng=0.0,
               ncut=dim
)

n_states = 4

H_sys = Qobj(transmon.hamiltonian(energy_esys=True)[0:n_states,0:n_states])

drive_op = Qobj(transmon.n_operator(energy_esys=True)[0:n_states,0:n_states])

wq = H_sys.eigenenergies()[1]-H_sys.eigenenergies()[0]

N_rep = 10
N_fock = 20

num_A = 80

g = 0.130/34
kappa = 0.002/34