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
dim = 2

num_A = 300

g = 0.01
kappa = 0.002

fname = 'data/params/charge_qubit.npz'
np.savez(fname, drive_op=drive_op.full(), wq=wq, H_sys=H_sys.full(), dim=dim, num_A=num_A, g=g, kappa=kappa)