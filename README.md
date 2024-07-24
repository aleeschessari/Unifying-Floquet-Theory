The data used to generate the figure in the paper are available in the data/ folder and may be plotted with the notebooks in plot/test. This repo uses QuTip 5.0.1 and cupy. if you do not have cupy you can import functions.function_nocuda instead of functions.function in the notebooks

Instructions to generate the SNR data 
- First, you need to generate the g_parallel data. This generates the g_parallel and the <\tilde \sigma_z(0)>, that are used in the analytical formula for <a(t)>
- Then, you need to run SNR_dispersive_params_analytics, this generates the analytical <a(t)> and the values of A_r that generates a long-time dispersive SNR equals to the longitudinal one
- Finally, you can run the notebook SNR

If you ever use this code, or part of it, please cite https://arxiv.org/abs/2407.03417
