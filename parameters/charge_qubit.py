from qutip import destroy, qeye
import numpy as np

sm = destroy(2)
sp = destroy(2).dag()
sz = 2*(destroy(2).dag()*destroy(2)-1/2*qeye(2))
sy = -1j*(sm.dag()-sm)
sx = sm+sm.dag()

drive_op = sx

wq = 1

H_sys = wq/2 * sz

n_states = 2

N_rep = 10 # this means that we will have 2*N_rep+1 replicas
N_fock = 30

num_A = 40

g = 0.01
kappa = 0.002