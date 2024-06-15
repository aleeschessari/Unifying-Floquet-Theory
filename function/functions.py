#!/usr/bin/env python
# coding: utf-8

from qutip import *
import numpy as np

def diag_replica(N_rep, target_replica):
    N_replica_space = 2*N_rep+1 # should be odd       
    diag = np.zeros(N_replica_space)
    indices = np.arange(-N_rep,N_rep+1)
    index_replica = np.where(indices == target_replica)
    diag[index_replica] = 1
    return np.diag(diag)

def project_replica(N_rep, target_replica, H_sys):
    # Projector over replica
    N_replica_space = 2*N_rep+1 # should be odd       
    diag = diag_replica(N_rep,target_replica)
    projector_rep = diag
    return tensor(Qobj(projector_rep),qeye(H_sys.dims[0]))

def construct_eigenstates_nofloquet(H_sys, n_states):

    evals, ekets = H_sys.eigenstates()
        
    return evals[0:n_states], ekets[0:n_states]

def construct_eigenstates_nofloquet_static_detuning(H_sys, drive_op, e0_list):
    evals_list = []
    ekets_list = []
    for e0 in e0_list:
        H = H_sys + e0/2*drive_op
        evals, ekets = H.eigenstates()
        evals_list.append(evals)
        ekets_list.append(ekets)

    return np.array(evals_list), ekets_list

def construct_eigenvalues_eigenstates_floquet(N_rep,u,v,H_sys,drive_op,w_d,n_states,option,target_replica):
    N_replica_space = 2*N_rep+1 # should be odd       

    sp_floquet = tensor(Qobj(np.diag(np.ones(N_replica_space-1),1)))
    sm_floquet = tensor(Qobj(np.diag(np.ones(N_replica_space-1),-1)))
    
    start = -(N_replica_space-1)/2*w_d
    stop = (N_replica_space-1)/2*w_d
    
    if w_d != 0:
        ll = np.linspace(start, stop, N_replica_space, endpoint=True)
    else:
        ll = np.zeros(N_replica_space)
    m = np.diag(ll)
    
    H_floquet = tensor(Qobj(np.diag(np.ones(N_replica_space))),H_sys)\
                +(u+1j*v)/2*tensor(sp_floquet,drive_op)\
                +(u-1j*v)/2*tensor(sm_floquet,drive_op)\
                +tensor(Qobj(m),qeye(H_sys.dims[0]))

    evals, ekets = H_floquet.eigenstates()
    weight = np.zeros(len(ekets))
    
    if option == 'reduced':
        for i,eigv in enumerate(ekets):
            weight[i] = (project_replica(N_rep,target_replica,H_sys) * eigv ).norm()

        order = np.argsort(weight)[::-1]

        evals = evals[order]
        ekets = ekets[order]

        temp_list_eigvalues = evals[0:n_states]
        temp_list_eigvectors = ekets[0:n_states]

        order2 = np.argsort(temp_list_eigvalues)[::1]

        temp_list_eigvalues = np.array(temp_list_eigvalues)
        temp_list_eigvectors = np.array(temp_list_eigvectors)

        evals = temp_list_eigvalues[order2]
        ekets = temp_list_eigvectors[order2]

        evals = [evals[0:n_states]]
        ekets = [ekets[i] for i in range(n_states)]
                        
    return evals, ekets, weight

def construct_eigenvalues_eigenstates_floquet_list(N_rep,last_A,num_A,H_sys,drive_op,w_d,n_states,target_replica):
    N_replica_space = 2*N_rep+1 # should be odd       

    sp_floquet = tensor(Qobj(np.diag(np.ones(N_replica_space-1),1)))
    sm_floquet = tensor(Qobj(np.diag(np.ones(N_replica_space-1),-1)))
    
    start = -(N_replica_space-1)/2*w_d
    stop = (N_replica_space-1)/2*w_d
    
    if w_d != 0:
        ll = np.linspace(start, stop, N_replica_space, endpoint=True)
    else:
        ll = np.zeros(N_replica_space)
    m = np.diag(ll)

    if isinstance(last_A, list):
        u_list = last_A
        num_A = len(last_A)
    elif isinstance(last_A, np.ndarray):
        u_list = last_A
        num_A = len(last_A)
    else:
        u_list = np.linspace(0,last_A,num = num_A)
    
    evals_list = np.zeros((len(u_list),n_states))
    evecs_list = []

    for idx,u in enumerate(u_list):
    
        H_qubit = tensor(Qobj(np.diag(np.ones(N_replica_space))),H_sys) + (u)/2*tensor(sp_floquet,drive_op) + (u)/2*tensor(sm_floquet,drive_op)\
                    +tensor(Qobj(m),qeye(H_sys.dims[0]))
    
        evals, ekets = H_qubit.eigenstates()
                    
        if u == 0:
            temp = construct_eigenvalues_eigenstates_floquet(N_rep,0,0,H_sys,drive_op,w_d,n_states,'reduced',target_replica)[1]
            evecs_list.append(temp)
            
        i_max = np.zeros((n_states), dtype=int)
        
        for i,temp_eigv in enumerate(temp):
            weight = 0
            for j,eigv in enumerate(ekets):
                if abs(temp_eigv.overlap(eigv)) > weight:
                    weight = abs(temp_eigv.overlap(eigv))
                    evals_list[idx,i] = evals[j] 
                    i_max[i] = j
                
        if u != 0:
            temp = []
            for i in range(len(i_max)):
                temp.append(ekets[i_max[i]])
            evecs_list.append(temp)

    return u_list, evals_list, evecs_list

def get_derivatives(N_rep,A_q,H_sys,drive_op,wd_list,n_states,num_A):

    if isinstance(A_q, list) or isinstance(A_q, np.ndarray):
        Z = np.zeros((len(wd_list),len(A_q),n_states))
        for i, w_d in enumerate(wd_list):
            Aq_list, evals = construct_eigenvalues_eigenstates_floquet_list(N_rep,A_q,num_A,H_sys,drive_op,w_d,n_states,0)[0], construct_eigenvalues_eigenstates_floquet_list(N_rep,A_q,num_A,H_sys,drive_op,w_d,n_states,0)[1]
            Z[i,:,:] = evals
    else:
        Z = np.zeros((len(wd_list),num_A,n_states))
        for i, w_d in enumerate(wd_list):
            Aq_list, evals = construct_eigenvalues_eigenstates_floquet_list(N_rep,A_q+0.1*A_q,num_A,H_sys,drive_op,w_d,n_states,0)[0], construct_eigenvalues_eigenstates_floquet_list(N_rep,A_q+0.1*A_q,num_A,H_sys,drive_op,w_d,n_states,0)[1]
            Z[i,:,:] = evals

    dd_real = np.gradient(Z, Aq_list, axis=1)
    dd2_real = np.gradient(dd_real, Aq_list, axis=1)
            
    return Aq_list, dd_real, dd2_real

def search_optimal_dispersive(N_rep,A_q,H_sys,g,drive_op,wq,wlist,n_states,num_A,ground,excited,kappa):
    dd2_real_disp = get_derivatives(N_rep,A_q,H_sys,drive_op,wlist,n_states,num_A)[2]

    chi1_disp = g**2*dd2_real_disp[:,2,excited]
    chi0_disp = g**2*dd2_real_disp[:,2,ground]

    chi_disp = chi1_disp-chi0_disp

    index_wr_low = np.argwhere(wlist > 0.3*wq)[0][0] \
        + (np.abs(np.abs(chi_disp[np.argwhere(wlist > 0.3*wq)[0][0]:np.argwhere(wlist >= wq)[0][0]])-kappa/2)).argmin() # find optimal dispersive readout low freq.

    index_wr_high = np.argwhere(wlist >= wq)[0][0] \
        + (np.abs(np.abs(chi_disp[np.argwhere(wlist >= wq)[0][0]:-1])-kappa/2)).argmin() # find optimal dispersive readout high freq.

    return chi_disp, index_wr_low, index_wr_high

def real_time_dynamics(H_sys,A_q,A_d,w_r,w_d,phi,g,drive_op,n_states,kappa,qubit_state,tlist,N_fock):
    psi0 = tensor(basis(N_fock,0), construct_eigenstates_nofloquet(H_sys, n_states)[1][qubit_state])
    
    a  = tensor(destroy(N_fock), qeye(H_sys.dims[0]))

    if A_q != 0:
        H_qubit_drive = A_q/2*tensor(qeye(N_fock),drive_op)
    else:
        H_qubit_drive = None

    H_coupling_1 = g*(a)*tensor(qeye(N_fock),drive_op)
    H_coupling_2 = g*(a.dag())*tensor(qeye(N_fock),drive_op)
    
    if A_d != 0:
        H_add_drive_stat = A_d/2*(a*np.exp(1j*phi)+a.dag()*np.exp(-1j*phi))
        H_add_drive_1 = A_d/2*(a*np.exp(-1j*phi))
        H_add_drive_2 = A_d/2*(a.dag()*np.exp(1j*phi))
    else:
        H_add_drive_stat = None
        H_add_drive_1 = None
        H_add_drive_2 = None

    if w_r - w_d != 0:
        H_extra = (w_r-w_d)*a.dag()*a
    else:
        H_extra = None

    def H1_coeff(t, args):
        return np.exp(-1j*w_d*t)

    def H2_coeff(t, args):
        return np.exp(1j*w_d*t)

    def H3_coeff(t, args):
        return np.exp(-1j*2*w_d*t)

    def H4_coeff(t, args):
        return np.exp(1j*2*w_d*t)
        
    H0 = [tensor(qeye(N_fock),H_sys),\
        [H_qubit_drive,H1_coeff],[H_qubit_drive,H2_coeff],\
        [H_coupling_1,H1_coeff],[H_coupling_2,H2_coeff],\
        H_add_drive_stat,[H_add_drive_1,H3_coeff],[H_add_drive_2,H4_coeff],\
        H_extra]
    
    H = []
    
    for elem in H0:
        if isinstance(elem, list):
            if elem[0] != None: 
                H.append([elem[0],elem[1]])
        else:
            if elem != None: 
                H.append(elem)
    
    def f(t, state):
        return expect(a, state)
    
    c_op_list = []
    
    c_op_list.append(np.sqrt(kappa)*a)    
    
    output = mesolve(H, psi0, tlist, c_op_list, [], options = Options(nsteps=40000))
    res = [f(tlist[i],output.states[i]) for i in range(len(tlist))]
    
    return res

def compute_weights(N_rep, ref, evecs):
    N_replica_space = 2*N_rep+1 # should be odd       

    renormalization = np.zeros(len(evecs), dtype = np.float64)
    
    weight = np.zeros((len(evecs),len(ref)), dtype = np.complex128)

    for i,eigv in enumerate(evecs):

        if N_replica_space > 1:
            for j in range(N_replica_space):
                for k in range(len(ref)):
                    weight[i,k] += np.array(tensor(basis(N_replica_space,j),ref[k]).overlap(eigv)) # how much i-th eigv is in ref k
        else:
            for k in range(len(ref)):
                weight[i,k] += np.array(ref[k].overlap(eigv)) # how much i-th eigv is in ref k

        for k in range(len(ref)):
            renormalization[i] += np.linalg.norm(weight[i,k])**2

            weight[i,k] = np.linalg.norm(weight[i,k])**2

    return weight, renormalization


def get_z0(N_rep,A_q,num_A,H_sys,drive_op,w_d,qubit_state,ground,excited,n_states):
    A_list, evals, evecs = construct_eigenvalues_eigenstates_floquet_list(N_rep,A_q,num_A,H_sys,drive_op,w_d,n_states,0)

    ref = construct_eigenstates_nofloquet(H_sys, n_states)[1]

    index_A = np.abs(A_list-A_q).argmin()

    z0 = (np.abs(compute_weights(N_rep, ref, evecs[index_A])[0][qubit_state,excited])-np.abs(compute_weights(N_rep, ref, evecs[index_A])[0][qubit_state,ground]))/compute_weights(N_rep, ref, evecs[index_A])[1][ground]
    
    return z0

def analytical_time_dynamics(z0,w_r,w_d,A_d,phi,g_parallel,g_sum,chi,chi_sum,kappa,gamma,tlist):
    
    alpha = -1j*((g_sum-g_parallel+A_d/2*np.exp(-1j*phi))*((gamma+2*1j*chi)/(gamma+1j*chi))*(1-np.exp((-kappa/2-1j*(chi_sum-chi)-1j*(w_r-w_d))*tlist))/(1j*(chi_sum-chi)+1j*(w_r-w_d)+kappa/2)+(z0+1)*(g_parallel-1j*chi/(gamma+1j*chi)*(g_sum+A_d/2*np.exp(-1j*phi)))/(1j*(chi_sum-chi)+1j*(w_r-w_d)+kappa/2-gamma)*(np.exp(-gamma*tlist)-np.exp((-kappa/2-1j*(chi_sum-chi)-1j*(w_r-w_d))*tlist)))
    
    beta = -1j*(g_sum+g_parallel+A_d/2*np.exp(-1j*phi))*(np.exp(-gamma*tlist)*(z0+1)*(1-np.exp((-kappa/2-1j*(chi_sum+chi)-1j*(w_r-w_d))*tlist))/(1j*(chi_sum+chi)+1j*(w_r-w_d)+kappa/2))
    
    return 1/(gamma+2*1j*chi)*((gamma+1j*chi)*alpha+1j*chi*beta)

def generate_SNR_list(res_a,kappa,t_list):

    SNR = []
    
    for idx,tau in enumerate(t_list):
        diff = np.abs(np.array(res_a[0][0])-np.array(res_a[1][0]))**2            
        SNR.append(np.sqrt(2*kappa*np.trapz(diff[0:idx], x=t_list[0:idx])))
    
    return SNR

