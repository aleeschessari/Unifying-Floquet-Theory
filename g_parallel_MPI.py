import function.functions_nocuda as functions

from qutip import *
import numpy as np
from mpi4py import MPI

# Set device and parameters in this part

############################# Set parameters ##############################
device = 'fluxonium' # change here to change the device: charge_qubit, flopping_spin, flopping_charge, transmon, fluxonium

data = np.load('data/params/'+device+'.npz', allow_pickle=True)
H_sys, drive_op, wq, g, kappa, num_A, dim = data['H_sys'], data['drive_op'], data['wq'], data['g'], data['kappa'], data['num_A'], data['dim']
#############################################################################

########################### Parameters of the paper ###########################
if device == 'charge_qubit':
    num_w = 151

    wlist = np.linspace(0,2,num_w,endpoint=True)
    Alist = [0.05*wq,0.5*wq,2.45*wq]
    
    n_states = 2
    N_rep =  60 # this means that we will have 2*N_rep+1 replicas
    
    ground, excited = 0, 1

    num_A = 300

elif device == 'flopping_spin':
    num_w = 101

    wlist = np.linspace(0,2*wq,num_w,endpoint=True)
    Alist = [0.2*wq, 0.3*wq]
    
    n_states = 4
    N_rep =  15 # this means that we will have 2*N_rep+1 replicas

    ground, excited = 0, 1

elif device == 'transmon':
    num_w = 101

    wlist = np.linspace(0,2*wq,101,endpoint=True)
    Alist = [0.037*wq, 0.09*wq]

    N_rep =  25 # this means that we will have 2*N_rep+1 replicas
    n_states = 25
    
    Ejeff, EC = data['Ejeff'], data['EC']
    
    ground, excited = 0, 1
    
elif device == 'fluxonium':
    num_w = 101

    wlist = np.linspace(0,3*wq,101,endpoint=True)
    Alist = [0.6*wq, 1.2*wq]

    N_rep =  15 # this means that we will have 2*N_rep+1 replicas

    ground, excited = 0, 1

    n_states = 10

else:
    print('select a valid device')
    sys.exit()

qubit_state_list = [ground, excited]
################################################################################

########################### Custom parameters #################################
# test other parameters A_q, ground, excited, w_r, w_r_disp, compensation

save_file = True # test mode or save data to generate the data of the figure of the paper
################################################################################
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

num_per_rank = len(wlist) // size # the floor division // rounds the result down to the nearest whole number.

lower_bound = rank * num_per_rank
upper_bound = (rank + 1) * num_per_rank

if rank == size-1:
    upper_bound = len(wlist)

print("This is processor ", rank, "and I am summing numbers from", lower_bound," to ", upper_bound - 1, flush=True)

wlist = wlist[lower_bound:upper_bound]

H_sys = Qobj(H_sys[0:n_states,0:n_states])
drive_op = Qobj(drive_op[0:n_states,0:n_states])

if save_file:
    fname = 'data/'+device+'/g_parallel/N_rep='+str(N_rep)+'_n_states='+str(n_states)\
            +'_dim='+str(dim)+'_num_w='+str(num_w)

    if rank == 0:
        data = open(fname+'.txt',"w")
        data.close()
        
else:
    fname = None

g_parallel = np.zeros((len(wlist), len(Alist)), dtype=np.complex128)
g_parallel_0 = np.zeros((len(wlist), len(Alist)))

A_list, dd_real, dd2_real, z0_list = functions.get_derivatives(N_rep,Alist,H_sys,drive_op,wlist,n_states,num_A,fname,True,qubit_state_list)

for i,A_q in enumerate(Alist):

    index_A = np.abs(A_list[0:(num_A-2)]-A_q).argmin()

    message = 'derivatives of the spectrum computed in A_q/w_q='+str(A_list[index_A]/wq)
    print(message)
    
    with open(fname+'.txt',"a") as data:
        data.write(message+"\n")
    
    g_parallel[:,i] = 1/2*g*(dd_real[:,index_A,1]-dd_real[:,index_A,0])

for i,A_q in enumerate(Alist):

    if device == 'charge_qubit':
        g_parallel_0[:, i] = g*A_q*wq*np.ones(len(wlist))/((wq*np.ones(len(wlist)))**2-wlist**2)

    elif device == 'transmon':
        n01_modulo_squared = 1/2*np.sqrt(Ejeff/(8*EC))
        
        evals_analytical = np.zeros((len(wlist), n_states))
        evals_floquet_analytical = np.zeros((len(wlist), 2))

        for j in range(n_states):
            evals_analytical[:, j] = (-Ejeff + np.sqrt(8*Ejeff*EC)*(j+1/2) - EC/12*(6*j**2 + 6*j + 3))*np.ones(len(wlist))
            
        evals_floquet_analytical[:, 0] = evals_analytical[:,0] \
            - n01_modulo_squared*wq*np.ones(len(wlist))/(wq**2*np.ones(len(wlist)) - wlist**2)
        evals_floquet_analytical[:, 1] = evals_analytical[:,1] \
            + n01_modulo_squared*(wq*np.ones(len(wlist))/(wq**2*np.ones(len(wlist)) - wlist**2)*\
                -2*(wq-EC)*np.ones(len(wlist))/((wq-EC)**2*np.ones(len(wlist))-wlist**2))

        g_parallel_0[:, i] = g*A_q/2*(n01_modulo_squared*(wq*np.ones(len(wlist))/(wq**2*np.ones(len(wlist)) - wlist**2)-2\
            *(wq-EC)*np.ones(len(wlist))/((wq-EC)**2*np.ones(len(wlist))-wlist**2))\
                +n01_modulo_squared*wq*np.ones(len(wlist))/(wq**2*np.ones(len(wlist)) - wlist**2))

if rank == 0:
    for i in range(size):
        if i != 0:
            g_parallel = np.concatenate((g_parallel, comm.recv(source=i, tag=i)), axis=0)
            g_parallel_0 = np.concatenate((g_parallel_0, comm.recv(source=i, tag=50+i)), axis=0)
            dd_real = np.concatenate((dd_real, comm.recv(source=i, tag=100+i)), axis=0)
            dd2_real = np.concatenate((dd2_real, comm.recv(source=i, tag=150+i)), axis=0)
            z0_list = np.concatenate((z0_list, comm.recv(source=i, tag=200+i)), axis=0)
            wlist = np.concatenate((wlist, comm.recv(source=i, tag=250+i)), axis=0)

    if save_file:
        np.savez(fname, Alist=Alist, wlist=wlist, g=g, A_list=A_list, dd_real=dd_real, dd2_real=dd2_real,\
            g_parallel=g_parallel, g_parallel_0=g_parallel_0, z0_list=z0_list)

else:
    comm.send(g_parallel, tag=rank, dest=0)
    comm.send(g_parallel_0, tag=50+rank, dest=0)
    comm.send(dd_real, tag=100+rank, dest=0)
    comm.send(dd2_real, tag=150+rank, dest=0)
    comm.send(z0_list, tag=200+rank, dest=0)
    comm.send(wlist, tag=250+rank, dest=0)