from qutip import *
import numpy as np

delta = 1
phi=0.9
tsc = np.cos(phi)/0.6
tsf = np.sin(phi)/0.6
epsilon = 0

sm = tensor(qeye(2),destroy(2))
sp = tensor(qeye(2),destroy(2).dag())
sz = tensor(qeye(2),2*(destroy(2).dag()*destroy(2)-1/2*qeye(2)))
sy = -1j*(sm.dag()-sm)
sx = sm+sm.dag()

tm = tensor(destroy(2),qeye(2))
tp = tensor(destroy(2).dag(),qeye(2))
tz = tensor(2*(destroy(2).dag()*destroy(2)-1/2*qeye(2)) ,qeye(2))
ty = -1j*(tm.dag()-tm)
tx = tm+tm.dag()

H_sys = epsilon/2 * tz + delta/2 * sz + tsc * tx - tsf * ty * sy        
H_sys_notsc_notsf = epsilon/2 * tz + delta/2 * sz        
H_sys_notsf = epsilon/2 * tz + delta/2 * sz + tsc * tx 

H_ref = 1/2 * tz - 1/2 * sz

drive_op = tz

wq = H_sys.eigenenergies()[1]-H_sys.eigenenergies()[0]

n_states = 4
dim = 4

num_A = 100

g  = 0.02 
kappa = 0.002

H_sys = Qobj(H_sys.full())

fname = 'data/params/flopping_spin.npz'
np.savez(fname, drive_op=drive_op.full(), wq=wq, H_sys=H_sys.full(), H_sys_notsc_notsf=H_sys_notsc_notsf.full(),\
    H_sys_notsf=H_sys_notsf.full(), H_ref=H_ref.full(),\
    dim=dim, num_A=num_A, g=g, kappa=kappa)