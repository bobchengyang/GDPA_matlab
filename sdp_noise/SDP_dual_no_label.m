clear all;
clc;
close all;

% cvx_solver SDPT3 % SDPT3 sometimes fails but always balanced; SeDuMi's may not be balanced!!!
cvx_solver SeDuMi % SDPT3 sometimes fails but always balanced; SeDuMi's may not be balanced!!!
cvx_begin quiet

%%=======change the following things for experiments======
[dataset_str,read_data] = get_data();
n_sample=round(0.05*size(read_data,1)); % total number of samples to be experimented
mu=1; % GSP label assignment weight after Eq.(10)
%%========================================================

[cL,label] = dataLoader_normalized(read_data,n_sample);
b_ind = 1:1:round((n_sample/n_sample^2)); % every other sample / ~50% training sample
disp(['True labels: ' num2str(label')]);

%% competing scheme 1: GLR quadratic closed-form
disp('GLR quadratic results====================================================================');
[x_pred_GLR_quad,error_count_GLR_quad,obj_GLR_quad] = GLR_closed_form(...
    cL,...
    n_sample,...
    b_ind,...
    label);

disp(['GLR quadratic predicted labels: ' num2str(x_pred_GLR_quad')]);
disp(['GLR quadratic error_count: ' num2str(error_count_GLR_quad)]);
disp(['GLR quadratic obj: ' num2str(obj_GLR_quad)]);

%% competing scheme 2: SDP ADMM


%% competing scheme 3: LP approach to SDP


%% Eq. (5) min_{x,X} tr(cL*X) SDP (interior-point)
% disp('Eq.(5) results====================================================================');
% [x_pred_eq5,error_count_eq5,obj_eq5,]=eq5(cL,n_sample,b_ind,label);
% disp(['Eq.(5) predicted labels: ' num2str(int8(x_pred_eq5'))]);
% disp(['Eq.(5) error_count: ' num2str(error_count_eq5)]);
% disp(['Eq.(5) obj: ' num2str(obj_eq5)]);

%% Eq. (8) max_{M} sum(vec(L.*M)) SDP standard form (interior-point)
disp('Eq.(8) results====================================================================');
dL = [-cL zeros(n_sample,1); zeros(1,n_sample) 0]; % eq. (6)
ei_s = eye(n_sample);
for i=1:length(b_ind)
    dB(:,:,i) = [zeros(n_sample,n_sample) ei_s(:,b_ind(i)); ei_s(b_ind(i),:) 0]; % eq. (7), where dB is a stacked matrix
end
[x_pred_eq8,error_count_eq8,obj_eq8,t_orig_end_eq8] = eq8_no_label(dL,n_sample,b_ind,label);

disp(['Eq.(8) predicted labels: ' num2str(int8(x_pred_eq8'))]);
disp(['Eq.(8) error_count: ' num2str(error_count_eq8)]);
disp(['Eq.(8) obj: ' num2str(obj_eq8)]);
disp(['Eq.(8) run-time: ' num2str(t_orig_end_eq8) 's']);

%% Eq. (10) min_{y,z} sum(y)+sum(b.*z) SDP-dual (interior-point)
disp('Eq.(10) results====================================================================');
ei_l = eye(n_sample+1);   %% two identity matrices
for i=1:n_sample+1
    dA(:,:,i) = diag(ei_l(:,i)); % eq. (7), where dA is a stacked matrix where each of them has only 1 non-zero entry
end
[y_eq10,z_eq10,obj_eq10,db,x_pred_eq10,error_count_eq10,t_orig_end_eq10] = eq10_no_label(label,b_ind,n_sample,dA,dL);
disp(['Eq.(10) predicted labels: ' num2str(int8(x_pred_eq10'))]);
disp(['Eq.(10) error_count: ' num2str(error_count_eq10)]);
disp(['Eq.(10) y: ' num2str(vec(y_eq10)')]);
disp(['Eq.(10) z: ' num2str(vec(z_eq10)')]);
disp(['Eq.(10) obj: ' num2str(obj_eq10)]);
disp(['Eq.(10) run-time: ' num2str(t_orig_end_eq10) 's']);

disp('Eq. no label====================================================================');
[y_new_enl,current_obj_enl,x_pred_enl,err_count_enl,t_orig_end_eq_no_label] = eq_no_label(label,n_sample,cL);
disp(['Eq. no label predicted labels: ' num2str(int8(x_pred_enl'))]);
disp(['Eq. no label error_count: ' num2str(err_count_enl)]);
disp(['Eq. no label y: ' num2str(vec(y_new_enl)')]);
disp(['Eq. no label obj: ' num2str(current_obj_enl)]);
disp(['Eq. no label run-time: ' num2str(t_orig_end_eq_no_label) 's']);





