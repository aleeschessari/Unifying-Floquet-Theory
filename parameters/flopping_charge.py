from qutip import tensor, qeye, destroy
import numpy as np

delta = 1
phi=np.pi/1.4
tsc = np.cos(phi)/1.8
tsf = np.sin(phi)/1.8
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

N_rep = 10
N_fock = 30

n_states = 4

num_A = 40

g  = 0.01 
kappa = 0.002