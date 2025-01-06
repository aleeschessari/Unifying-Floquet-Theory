import function.functions_nocuda as functions

from qutip import *
import numpy as np
import importlib
from mpi4py import MPI

# Set device and parameters in this part

################################ Select device ################################
device = 'fluxonium' # change here to change the device: charge_qubit, flopping_spin, flopping_charge, transmon, fluxonium

data = np.load('data/params/'+device+'.npz', allow_pickle=True)
H_sys, drive_op, wq, g, kappa, num_A, dim = data['H_sys'], data['drive_op'], data['wq'], data['g'], data['kappa'], data['num_A'], data['dim']
###############################################################################

########################### Parameters of the paper ###########################
if device == 'charge_qubit':
    num_w = 151
    
    N_rep_import = 60
    n_states_import = 2
    
    n_states = 2
    N_fock = 8

    proj = None

elif device == 'flopping_spin':
    num_w = 101
    
    N_rep_import = 15
    n_states_import = 4
    
    n_states = 4
    N_fock = 6

    proj = None

elif device == 'transmon':
    num_w = 101
    
    N_rep_import = 10
    n_states_import = 15
    
    N_fock = 14
    n_states = 15

    cutoff = 4
    proj = Qobj(np.diag(np.concatenate((np.zeros(n_states-cutoff),np.ones(cutoff)))))

elif device == 'fluxonium':
    num_w = 101
    
    N_rep_import = 10
    n_states_import = 20

    N_fock = 8
    n_states = 10

    cutoff = 4
    proj = Qobj(np.diag(np.concatenate((np.zeros(n_states-cutoff),np.ones(cutoff)))))

else:
    print('select a valid device')
    sys.exit()

final_t = 0.5

tlist = np.linspace(0,final_t/kappa,num=200)

ground, excited = 0, 1

compensation = False # set compensation True/False

H_sys = Qobj(H_sys[0:n_states,0:n_states]) # truncation
drive_op = Qobj(drive_op[0:n_states,0:n_states]) # truncation
###########################################################################

########################## Import data ####################################
fname_import ='data/'+device+'/SNR_params_and_analytics_N_rep='+str(N_rep_import)+'_n_states='+str(n_states_import)+'_dim='+str(dim)+'_num_w='\
    +str(num_w)+'_final_t='+str(final_t)+'_compensation='+str(compensation)

def import_npz(npz_file):
    Data = np.load(npz_file, allow_pickle=True)
    for varName in Data:
        globals()[varName] = Data[varName]

import_npz(fname_import+'.npz')
###############################################################################

########################### Custom parameters #################################
# test other parameters wlist, Alist, compensation

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

if save_file:
    fname = 'data/'+device+'/SNR'+'_N_fock='+str(N_fock)+'_N_rep_import='+str(N_rep_import)+'_n_states='+str(n_states)\
            +'_dim='+str(dim)+'_num_w='+str(num_w)+'_final_t='+str(final_t)+'_compensation='+str(compensation)

    if rank == 0:
        data = open(fname+'.txt', "w")
        data.close()

        data = open(fname+'_disp.txt', "w")
        data.close()

print("This is processor ", rank, "and I am summing numbers from", lower_bound," to ", upper_bound - 1, flush=True)

A_d_array = A_d_array[:,lower_bound:upper_bound]
wlist = wlist[lower_bound:upper_bound]
w_d_array = w_d_array[:,lower_bound:upper_bound]
A_r_array = A_r_array[:,lower_bound:upper_bound]
w_r_disp_array = w_r_disp_array[lower_bound:upper_bound]
w_d_disp_array = w_d_disp_array[lower_bound:upper_bound]

res_num = np.zeros((len(Alist),upper_bound-lower_bound), dtype=object)
res_num_disp = np.zeros((len(Alist),upper_bound-lower_bound), dtype=object)

for i, A_q in enumerate(Alist):    
    res_a_num, exp_fock, exp_proj = functions.real_time_dynamics(H_sys,A_q,A_d_array[i,:],wlist,w_d_array[i,:],0,g,drive_op,n_states,kappa,[ground,excited],tlist,N_fock,proj,fname)

    if i == 0:
        res_a_disp, exp_fock_disp, exp_proj_disp = functions.real_time_dynamics(H_sys,0,A_r_array[i,:],w_r_disp_array,w_d_disp_array,3/2*np.pi,g,drive_op,n_states,kappa,[ground,excited],tlist,N_fock,proj,fname+'_disp')

    for j, w_r in enumerate(wlist):

        res_num[i,j] = functions.generate_SNR_list(res_a_num[j,:],kappa,tlist)[-1]

        if i == 0:
            res_num_disp[i,j] = functions.generate_SNR_list(res_a_disp[j,:],kappa,tlist)[-1]


if rank == 0:
    for i in range(size):
        if i != 0:
            res_num = np.concatenate((res_num, comm.recv(source=i, tag=i)), axis=1)
            res_num_disp = np.concatenate((res_num_disp, comm.recv(source=i, tag=100+i)), axis=1)
            wlist = np.concatenate((wlist, comm.recv(source=i, tag=150+i)), axis=0)

    if save_file:
            np.savez(fname, Alist=Alist, wlist=wlist, res_an=res_an, res_num_disp=res_num_disp, res_num=res_num,\
                    index_wr_low=index_wr_low, index_wr_high=index_wr_high, chi_disp=chi_disp)
else:
    comm.send(res_num, tag=rank, dest=0)
    comm.send(res_num_disp, tag=100+rank, dest=0)
    comm.send(wlist, tag=150+rank, dest=0)