clear all;
clc;
close all;

% cvx_solver SDPT3 % SDPT3 sometimes fails but always balanced; SeDuMi's may not be balanced!!!
cvx_solver SeDuMi % SDPT3 sometimes fails but always balanced; SeDuMi's may not be balanced!!!
% cvx_begin quiet

%%=======change the following things for experiments======
[dataset_str,read_data] = get_data();
n_sample=round(0.1*size(read_data,1)); % total number of samples to be experimented
rho=1e-5; % PSD parameter
y_initial_scale=1e1; % for y initialization (LP variables in Eq. (12))
z_initial_scale=1e0; % for z initialization (LP variables in Eq. (12))
mu=1; % GSP label assignment weight after Eq.(10)
%%========================================================

[cL,label] = dataLoader_normalized(read_data,n_sample);
b_ind = 1:1:round((n_sample/1)); % every other sample / ~50% training sample
disp(['True labels: ' num2str(label(round((n_sample/2))+1:1:n_sample)')]);

%% Eq. (14) SDP dual
[eq14_sdp_obj] = eq11_sdp(label,b_ind,n_sample,cL);