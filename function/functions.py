#!/usr/bin/env python
# coding: utf-8

from qutip import *
import numpy as np
import cupyx.scipy as sp
import cupy as cp

def diag_replica(N_rep, target_replica):
    N_replica_space = 2*N_rep+1 # should be odd       
    diag = cp.zeros(N_replica_space)
    indices = cp.arange(-N_rep,N_rep+1)
    index_replica = cp.where(indices == target_replica)
    diag[index_replica] = 1
    return diag

def construct_eigenstates_nofloquet_noqutip(H_sys, n_states):

    evals, ekets = cp.linalg.eigh(cp.asarray(H_sys.full()))
        
    return evals[0:n_states], ekets[:,0:n_states]

def construct_eigenstates_nofloquet(H_sys, n_states):

    evals, ekets = H_sys.eigenstates()
        
    return evals[0:n_states], ekets[0:n_states]

def construct_eigenstates_nofloquet_static_detuning(H_sys, drive_op, e0_list):
    evals_list = []
    ekets_list = []
    for e0 in e0_list:
        H = H_sys + e0/2*drive_op
        evals, ekets = np.linalg.eigh(H.full())
        evals_list.append(evals)
        ekets_list.append(ekets)

    return np.array(evals_list), ekets_list

def construct_eigenvalues_eigenstates_floquet_list(N_rep, last_A, num_A,
                                                   H_sys, drive_op, w_d, 
                                                   n_states, target_replica, fname=None):
    N_replica_space = 2*N_rep+1 # should be odd       
    
    sp_floquet = cp.diag(np.ones(N_replica_space-1), 1)
    sm_floquet = cp.diag(np.ones(N_replica_space-1), -1)

    start = -(N_replica_space-1)/2*w_d
    stop = (N_replica_space-1)/2*w_d

    if w_d != 0:
        ll = cp.linspace(start, stop, N_replica_space, endpoint=True)
    else:
        ll = cp.zeros(N_replica_space)
   
    m = cp.diag(ll)
    
    tensor_0 = cp.kron(cp.identity(N_replica_space), cp.asarray(H_sys.full()))
    tensor_1 = cp.kron(sp_floquet, cp.asarray(drive_op.full()))
    tensor_2 = cp.kron(sm_floquet, cp.asarray(drive_op.full()))
    tensor_3 = cp.kron(m, cp.diag(cp.ones(n_states)))

    # Handling the exceptions for the amplitude of the drive
    if isinstance(last_A, list):
        u_list = last_A
        num_A = len(last_A)
    elif isinstance(last_A, np.ndarray):
        u_list = last_A
        num_A = len(last_A)
    else:
        u_list = np.linspace(0, last_A, num = num_A)

    evecs_list = []

    H0 = tensor_0 + tensor_3 
    H1 = 1/2*tensor_1 + 1/2*tensor_2

    message = ''

    evals_list = cp.zeros((len(u_list), n_states)) # Matrix that stores A vs E (given all the bands)

    evals, evecs = construct_eigenstates_nofloquet_noqutip(H_sys, n_states)
    temp = cp.kron(diag_replica(N_rep,target_replica), cp.transpose(cp.conjugate(evecs)))

    proj_last_replica = cp.kron(cp.diag(diag_replica(N_rep,N_rep)), cp.identity(n_states))\
                            +cp.kron(cp.diag(diag_replica(N_rep,-N_rep)), cp.identity(n_states))
    
    if n_states > 5:
        proj_last_states = cp.kron(cp.identity(N_replica_space), cp.diag(cp.concatenate((cp.zeros(n_states-4),cp.ones(4)))))

    for idx, u in enumerate(u_list):
        H_qubit = H0 + u*H1
        evals, ekets = cp.linalg.eigh(H_qubit)

        #partial_weight = cp.einsum('ij,ik->jk', cp.conjugate(ekets), temp)
        partial_weight = temp @ ekets # ekets has dim N_replica_space * N_replica_space. Eigenvectors are on the columns

        weight = partial_weight*cp.conjugate(partial_weight)                
        i_max = cp.argmax(weight, axis=1)
        evals_list[idx, :] = evals[i_max] 

        temp = cp.transpose(cp.conjugate(ekets[:,i_max]))
        evecs_list.append(ekets[:,i_max])

        if w_d > 0:
            overflow_replica = cp.linalg.norm(proj_last_replica @ ekets[:,i_max], axis=0)
            
            if overflow_replica[0] > 0.01:
                msg = 'not enough replicas to describe the ground state at A_q='+str(u)+' and w_d='+str(w_d)+', leakage='+str(np.round(overflow_replica[0],3))
                print(msg)
                message += msg+"\n"

            if overflow_replica[1] > 0.01:
                msg = 'not enough replicas to describe the first excited state at A_q='+str(u)+' and w_d='+str(w_d)+', leakage='+str(np.round(overflow_replica[1],3))
                print(msg)
                message += msg+"\n"

        if n_states > 5:
            overflow_states = cp.linalg.norm(proj_last_states @ ekets[:,i_max], axis=0)

            if overflow_states[0] > 0.01:
                msg = 'not enough states to describe the ground state at A_q='+str(u)+' and w_d='+str(w_d)+', leakage='+str(np.round(overflow_states[0],3))
                print(msg)
                message += msg+"\n"

            if overflow_states[1] > 0.01:
                msg = 'not enough states to describe the first excited state at A_q='+str(u)+' and w_d='+str(w_d)+', leakage='+str(np.round(overflow_states[1],3))
                print(msg)
                message += msg+"\n"

    if fname != None:
        with open(fname+'.txt',"a") as data:
            data.write(message)

    return u_list, evals_list, evecs_list

def get_derivatives(N_rep,A_q,H_sys,drive_op,wd_list,n_states,num_A,fname=None,z0_comput=None,qubit_state_list=None):

    if z0_comput != None:
        ref = construct_eigenstates_nofloquet_noqutip(H_sys, n_states)[1]

    if isinstance(A_q, np.ndarray):
        Z = cp.zeros((len(wd_list),len(A_q),n_states))
        z0 = cp.zeros((len(wd_list),len(A_q),2))

        for i, w_d in enumerate(wd_list):
            Aq_list, evals, evecs = construct_eigenvalues_eigenstates_floquet_list(N_rep,A_q,num_A,H_sys,drive_op,w_d,n_states,0,fname)
            if z0_comput != None:
                for ii, A in enumerate(A_q):
                    w, r = compute_weights(N_rep, ref, evecs[ii])
                    for idx, qubit_state in enumerate(qubit_state_list):
                        z0[i,ii,idx]=((w[qubit_state_list[1],qubit_state]-w[qubit_state_list[0],qubit_state])/r[qubit_state])

            Z[i,:,:] = evals

    elif isinstance(A_q, list):
        A_qf = A_q[-1]
        Z = cp.zeros((len(wd_list),num_A,n_states))

        z0 = cp.zeros((len(wd_list),len(A_q),2))

        for i, w_d in enumerate(wd_list):
            Aq_list, evals, evecs = construct_eigenvalues_eigenstates_floquet_list(N_rep,A_qf+0.1*A_qf,num_A,H_sys,drive_op,w_d,n_states,0,fname)
            if z0_comput != None:
                for ii, A in enumerate(A_q):
                    index_A = np.abs(Aq_list-A).argmin()
                    w, r = compute_weights(N_rep, ref, evecs[index_A])
                    for idx, qubit_state in enumerate(qubit_state_list):
                        z0[i,ii,idx]=((w[qubit_state_list[1],qubit_state]-w[qubit_state_list[0],qubit_state])/r[qubit_state])

            Z[i,:,:] = evals

    else:
        A_qf = A_q
        Z = cp.zeros((len(wd_list),num_A,n_states))

        z0 = cp.zeros((len(wd_list),1,2))

        for i, w_d in enumerate(wd_list):
            Aq_list, evals, evecs = construct_eigenvalues_eigenstates_floquet_list(N_rep,A_qf+0.1*A_qf,num_A,H_sys,drive_op,w_d,n_states,0,fname)
            if z0_comput != None:
                index_A = np.abs(Aq_list-A_qf).argmin()
                w, r = compute_weights(N_rep, ref, evecs[index_A])
                for idx, qubit_state in enumerate(qubit_state_list):
                    z0[i,0,idx]=((w[qubit_state_list[1],qubit_state]-w[qubit_state_list[0],qubit_state])/r[qubit_state])

            Z[i,:,:] = evals

    dd_real = cp.gradient(Z, Aq_list, axis=1)
    dd2_real = cp.gradient(dd_real, Aq_list, axis=1)
            
    if z0_comput != None:
        return Aq_list, dd_real.get(), dd2_real.get(), z0.get()
    else:
        return Aq_list, dd_real.get(), dd2_real.get()

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

def real_time_dynamics(H_sys,A_q,A_d_list,w_r_list,w_d_list,phi,g,drive_op,n_states,kappa,qubit_state_list,tlist,N_fock,proj=None,fname=None):
    
    a  = tensor(destroy(N_fock), qeye(H_sys.dims[0]))
        
    c_op_list = []
    c_op_list.append(np.sqrt(kappa)*a)    

    if proj != None:
        e_ops_list = [a, tensor(Qobj(np.diag(np.concatenate((np.zeros(N_fock-1), np.ones(1))))), qeye(H_sys.dims[0])), tensor(qeye(N_fock),proj)]
    else:
        e_ops_list = [a, tensor(Qobj(np.diag(np.concatenate((np.zeros(N_fock-1), np.ones(1))))), qeye(H_sys.dims[0]))]

    H1_coeff = 'exp(-1j*w_d*t)'
    H2_coeff = 'exp(1j*w_d*t)'
    H3_coeff = 'exp(-1j*2*w_d*t)'
    H4_coeff = 'exp(1j*2*w_d*t)'

    H_fictitious = 'A_d/2'

    H_coupling_1 = g*(a)*tensor(qeye(N_fock),drive_op)
    H_coupling_2 = g*(a.dag())*tensor(qeye(N_fock),drive_op)

    H_qubit_drive = A_q/2*tensor(qeye(N_fock),drive_op)

    H_add_drive_stat = (a*np.exp(1j*phi)+a.dag()*np.exp(-1j*phi))
    H_add_drive_1 = (a*np.exp(-1j*phi))
    H_add_drive_2 = (a.dag()*np.exp(1j*phi))

    H = [tensor(qeye(N_fock),H_sys),\
        [H_coupling_1,H1_coeff],[H_coupling_2,H2_coeff]]

    if A_q != 0:
        H.append([H_qubit_drive,H1_coeff])
        H.append([H_qubit_drive,H2_coeff])

    dict = {}
    dict['w_d'] = w_d_list[0]

    if A_d_list.any():
        H.append([H_add_drive_stat,H_fictitious])
        H.append([H_add_drive_1,H_fictitious+'*'+H3_coeff])
        H.append([H_add_drive_2,H_fictitious+'*'+H4_coeff])
        dict['A_d'] = A_d_list[0]
    if not np.array_equal(w_d_list, w_r_list):
        H_extra = a.dag()*a
        H.append([H_extra, 'w_rd'])
        dict['w_rd'] = w_r_list[0]-w_d_list[0]

    res = np.zeros((len(w_d_list), len(qubit_state_list)), dtype=object)
    res_proj = np.zeros((len(w_d_list), len(qubit_state_list)), dtype=object)
    res_fock = np.zeros((len(w_d_list), len(qubit_state_list)), dtype=object)

    solver = MCSolver(QobjEvo(H, args=dict), c_ops=c_op_list, options={"progress_bar": False}) # unraveling
    #solver = MESolver(QobjEvo(H, args=dict), c_ops=c_op_list) # exact solver for Lindblad equation

    message = ''

    for i, w_d in enumerate(w_d_list):
        dict['w_d'] = w_d

        if A_d_list.any():
            dict['A_d'] = A_d_list[i]
        if not np.array_equal(w_d_list, w_r_list):
            dict['w_rd'] = w_r_list[i]-w_d_list[i]

        for j, qubit_state in enumerate(qubit_state_list):

            psi0 = tensor(basis(N_fock,0), construct_eigenstates_nofloquet(H_sys, n_states)[1][qubit_state])

            output = solver.run(psi0, tlist, e_ops=e_ops_list, args=dict, ntraj=5) # average over ntraj

            res[i,j] = output.expect[0]
            res_fock[i,j] = output.expect[1]
            if np.max(res_fock[i,j])>10**(-2):
                msg = 'Attention : leaking out fock space at A_q='+str(A_q)+' and w_d='+str(w_d)+', leakage='+str(no.round(np.max(res_fock[i,j]),3))
                print(msg)
                message += msg+"\n"

            if proj != None:
                res_proj[i,j] = output.expect[2]
                if np.max(res_proj[i,j])>10**(-2):
                    msg = 'Attention : leaking out the cutoff at A_q='+str(A_q)+' and w_d='+str(w_d)+', leakage='+str(no.round(np.max(res_proj[i,j]),3))
                    print(msg)
                    message += msg+"\n"

    if fname != None: 
        with open(fname+'.txt',"a") as data:
            data.write(message)

    return res, res_fock, res_proj

def compute_weights(N_rep, ref, evecs):
    N_replica_space = 2*N_rep+1 # should be odd       
    
    temp_weight = cp.zeros((np.shape(ref)[1], np.shape(evecs)[1]), dtype = np.complex128)

    if N_replica_space > 1:
        for j in range(N_replica_space):
            temp = cp.zeros(N_replica_space)
            temp[j] = 1
            temp_weight += cp.kron(temp, cp.transpose(cp.conjugate(ref))) @ evecs # how much row-th ref is in column-th eigv
    
    # for the static spectrum
    else:
        temp_weight = cp.transpose(cp.conjugate(ref)) @ cp.asarray(evecs)  # how much row-th ref is in column-th eigv

    weight = cp.real(temp_weight*cp.conjugate(temp_weight))
    
    renormalization = cp.real(cp.sum(weight, axis=0))

    return weight.get(), renormalization.get()

def analytical_time_dynamics(z0,w_r,w_d,A_d,phi,g_parallel,g_sum,chi,chi_sum,kappa,gamma,tlist):
    
    alpha = -1j*((g_sum-g_parallel+A_d/2*np.exp(-1j*phi))*((gamma+2*1j*chi)/(gamma+1j*chi))*(1-np.exp((-kappa/2-1j*(chi_sum-chi)-1j*(w_r-w_d))*tlist))/(1j*(chi_sum-chi)+1j*(w_r-w_d)+kappa/2)+(z0+1)\
        *(g_parallel-1j*chi/(gamma+1j*chi)*(g_sum+A_d/2*np.exp(-1j*phi)))/(1j*(chi_sum-chi)+1j*(w_r-w_d)+kappa/2-gamma)*(np.exp(-gamma*tlist)-np.exp((-kappa/2-1j*(chi_sum-chi)-1j*(w_r-w_d))*tlist)))
    
    beta = -1j*(g_sum+g_parallel+A_d/2*np.exp(-1j*phi))*(np.exp(-gamma*tlist)*(z0+1)*(1-np.exp((-kappa/2-1j*(chi_sum+chi)-1j*(w_r-w_d))*tlist))/(1j*(chi_sum+chi)+1j*(w_r-w_d)+kappa/2))
    
    return (1/(gamma+2*1j*chi)*((gamma+1j*chi)*alpha+1j*chi*beta))

def generate_SNR_list(res_a,kappa,t_list):

    SNR = []
    
    for idx,tau in enumerate(t_list):
        diff = np.abs(np.array(np.ravel(res_a[1]))-np.array(np.ravel(res_a[0])))**2            
        SNR.append(np.sqrt(2*kappa*np.trapz(diff[0:idx], x=t_list[0:idx])))
    
    return SNR