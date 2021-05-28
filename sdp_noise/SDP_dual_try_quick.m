clear all;
clc;
close all;

% cvx_solver SDPT3 % SDPT3 sometimes fails but always balanced; SeDuMi's may not be balanced!!!
cvx_solver SeDuMi % SDPT3 sometimes fails but always balanced; SeDuMi's may not be balanced!!!
cvx_begin quiet

%%=======change the following things for experiments======
[dataset_str,read_data] = get_data();
n_sample=round(0.2*size(read_data,1)); % total number of samples to be experimented
rho=1e-5; % PSD parameter
y_initial_scale=1e1; % for y initialization (LP variables in Eq. (12))
z_initial_scale=1e0; % for z initialization (LP variables in Eq. (12))
mu=1; % GSP label assignment weight after Eq.(10)
%%========================================================

[cL,label] = dataLoader_normalized(read_data,n_sample);
b_ind = 1:1:round((n_sample/5)); % every other sample / ~50% training sample
disp(['True labels: ' num2str(label')]);

%% competing scheme 1: GLR quadratic closed-form
% disp('GLR quadratic results====================================================================');
% [x_pred_GLR_quad,error_count_GLR_quad,obj_GLR_quad] = GLR_closed_form(...
%     cL,...
%     n_sample,...
%     b_ind,...
%     label);
% 
% disp(['GLR quadratic predicted labels: ' num2str(x_pred_GLR_quad')]);
% disp(['GLR quadratic error_count: ' num2str(error_count_GLR_quad)]);
% disp(['GLR quadratic obj: ' num2str(obj_GLR_quad)]);

% [dy] = eq_slides(cL,n_sample,b_ind,label);


%% competing scheme 2: SDP ADMM


%% competing scheme 3: LP approach to SDP


%% Eq. (5) min_{x,X} tr(cL*X) SDP (interior-point)
disp('Eq.(5) results====================================================================');
[x_pred_eq5,error_count_eq5,obj_eq5]=eq5(cL,n_sample,b_ind,label);
disp(['Eq.(5) predicted labels: ' num2str(int8(x_pred_eq5'))]);
disp(['Eq.(5) error_count: ' num2str(error_count_eq5)]);
disp(['Eq.(5) obj: ' num2str(obj_eq5)]);

%% Eq. (5) min_{x,X} tr(cL*X) SDP (interior-point) no added rowcol
% disp('Eq.(5) no added rowcol results====================================================================');
% [x_pred_eq5_no_added_rowcol,error_count_eq5_no_added_rowcol,obj_eq5_no_added_rowcol]=eq5_no_added_rowcol(cL,n_sample,b_ind,label);
% disp(['Eq.(5) no added rowcol predicted labels: ' num2str(int8(x_pred_eq5_no_added_rowcol'))]);
% disp(['Eq.(5) no added rowcol error_count: ' num2str(error_count_eq5_no_added_rowcol)]);
% disp(['Eq.(5) no added rowcol obj: ' num2str(obj_eq5_no_added_rowcol)]);

% %% Eq. (5) balanced min_{x,X} tr(cL*X) SDP (interior-point)
% disp('Eq.(5) balanced results====================================================================');
% [x_pred_eq5_balanced,error_count_eq5_balanced,obj_eq5_balanced]=eq5_balanced(cL,n_sample,b_ind,label);
% disp(['Eq.(5) balanced predicted labels: ' num2str(x_pred_eq5_balanced')]);
% disp(['Eq.(5) balanced error_count: ' num2str(error_count_eq5_balanced)]);
% disp(['Eq.(5) balanced obj: ' num2str(obj_eq5_balanced)]);

%% Eq. (8) max_{M} sum(vec(L.*M)) SDP standard form (interior-point)
disp('Eq.(8) results====================================================================');
dL = [-cL zeros(n_sample,1); zeros(1,n_sample) 0]; % eq. (6)
ei_s = eye(n_sample);
for i=1:length(b_ind)
    dB(:,:,i) = [zeros(n_sample,n_sample) ei_s(:,b_ind(i)); ei_s(b_ind(i),:) 0]; % eq. (7), where dB is a stacked matrix
end
[x_pred_eq8,error_count_eq8,obj_eq8] = eq8(dL,dB,n_sample,b_ind,label);

disp(['Eq.(8) predicted labels: ' num2str(int8(x_pred_eq8'))]);
disp(['Eq.(8) error_count: ' num2str(error_count_eq8)]);
disp(['Eq.(8) obj: ' num2str(obj_eq8)]);

%% Eq. (8) max_{M} sum(vec(L.*M)) SDP standard form (interior-point) no added rowcol
% disp('Eq.(8) no added rowcol results====================================================================');
% % dL = [-cL zeros(n_sample,1); zeros(1,n_sample) 0]; % eq. (6)
% % ei_l = eye(n_sample+1);   %% two identity matrices
% ei_s = eye(n_sample);
% for i=1:length(b_ind)
%     if b_ind(i)==1
%     dB_no_added_rowcol(:,:,i) = zeros(n_sample); % eq. (7), where dB is a stacked matrix
%     dB_no_added_rowcol(1,1,i)=2;
%     else
%     ei_s_row=ei_s(b_ind(i),:);
%     ei_s_row(1)=[];
%     ei_s_col=ei_s(:,b_ind(i));
%     ei_s_col(1)=[];
%     dB_no_added_rowcol(:,:,i) = [0 ei_s_row; ei_s_col zeros(n_sample-1,n_sample-1)]; % eq. (7), where dB is a stacked matrix
%     end
% end
% [x_pred_eq8_no_added_rowcol,error_count_eq8_no_added_rowcol,obj_eq8_no_added_rowcol] = eq8_no_added_rowcol(-cL,dB_no_added_rowcol,n_sample,b_ind,label);
% 
% disp(['Eq.(8) no added rowcol predicted labels: ' num2str(int8(x_pred_eq8_no_added_rowcol'))]);
% disp(['Eq.(8) no added rowcol error_count: ' num2str(error_count_eq8_no_added_rowcol)]);
% disp(['Eq.(8) no added rowcol obj: ' num2str(obj_eq8_no_added_rowcol)]);

%% Eq. (10) min_{y,z} sum(y)+sum(b.*z) SDP-dual (interior-point)
disp('Eq.(10) results====================================================================');
ei_l = eye(n_sample+1);   %% two identity matrices
for i=1:n_sample+1
    dA(:,:,i) = diag(ei_l(:,i)); % eq. (7), where dA is a stacked matrix where each of them has only 1 non-zero entry
end
% [y_eq10,z_eq10,obj_eq10,db,x_pred_LA,error_count_LA,obj_LA] = eq10(label,b_ind,n_sample,dA,dB_no_added_rowcol,-cL,mu);
[y_eq10,z_eq10,obj_eq10,db,x_pred_eq10,error_count_eq10] = eq10(label,b_ind,n_sample,dA,dB,dL);
disp(['Eq.(10) predicted labels: ' num2str(int8(x_pred_eq10'))]);
disp(['Eq.(10) error_count: ' num2str(error_count_eq10)]);
disp(['Eq.(10) y: ' num2str(vec(y_eq10)')]);
disp(['Eq.(10) z: ' num2str(vec(z_eq10)')]);
disp(['Eq.(10) obj: ' num2str(obj_eq10)]);
% disp(['Eq.(10) then GSP LA predicted labels: ' num2str(x_pred_LA')]);
% disp(['Eq.(10) then GSP LA error_count: ' num2str(error_count_LA)]);
% disp(['Eq.(10) then GSP LA obj: ' num2str(obj_LA)]);

% [y_eq10_beta,z_eq10_beta,obj_eq10_beta,~,x_pred_eq10_beta,error_count_eq10_beta] = eq10_beta(label,b_ind,n_sample,dA,dB,dL);
% disp(['Eq.(10) beta predicted labels: ' num2str(int8(x_pred_eq10_beta'))]);
% disp(['Eq.(10) beta error_count: ' num2str(error_count_eq10_beta)]);
% disp(['Eq.(10) beta y: ' num2str(vec(y_eq10_beta)')]);
% disp(['Eq.(10) beta z: ' num2str(vec(z_eq10_beta)')]);
% disp(['Eq.(10) beta obj: ' num2str(obj_eq10_beta)]);

%% Eq. (10) min_{y,z} sum(y)+sum(b.*z) SDP-dual (interior-point) no added rowcol
% disp('Eq.(10) no added rowcol 1results====================================================================');
% clear dA
% ei_l = eye(n_sample);   %% two identity matrices
% for i=1:n_sample
%     dA(:,:,i) = diag(ei_l(:,i)); % eq. (7), where dA is a stacked matrix where each of them has only 1 non-zero entry
% end
% % [y_eq10,z_eq10,obj_eq10,db,x_pred_LA,error_count_LA,obj_LA] = eq10(label,b_ind,n_sample,dA,dB_no_added_rowcol,-cL,mu);
% [y_eq10_no_added_rowcol,z_eq10_no_added_rowcol,obj_eq10_no_added_rowcol,db,...
%     x_pred_eq10_no_added_rowcol,error_count_eq10_no_added_rowcol] = eq10_no_added_rowcol(label,b_ind,n_sample,dA,dB_no_added_rowcol,-cL);
% disp(['Eq.(10) no added rowcol predicted labels: ' num2str(int8(x_pred_eq10_no_added_rowcol'))]);
% disp(['Eq.(10) no added rowcol error_count: ' num2str(error_count_eq10_no_added_rowcol)]);
% disp(['Eq.(10) no added rowcol y: ' num2str(vec(y_eq10_no_added_rowcol)')]);
% disp(['Eq.(10) no added rowcol z: ' num2str(vec(z_eq10_no_added_rowcol)')]);
% disp(['Eq.(10) no added rowcol obj: ' num2str(obj_eq10_no_added_rowcol)]);
% disp(['Eq.(10) then GSP LA predicted labels: ' num2str(x_pred_LA')]);
% disp(['Eq.(10) then GSP LA error_count: ' num2str(error_count_LA)]);
% disp(['Eq.(10) then GSP LA obj: ' num2str(obj_LA)]);

%% Eq. (11) SDP dual
% disp('Eq.(11) N+2 results====================================================================');
% clear dA
% ei_l = eye(n_sample+1);   %% two identity matrices
% for i=1:n_sample+1
%     dA(:,:,i) = diag(ei_l(:,i)); % eq. (7), where dA is a stacked matrix where each of them has only 1 non-zero entry
% end
% [obj_eq11_otn,eg_eq11_otn] = eq11_sdp_original_to_new(label,b_ind,n_sample,dA,dB,dL);
% disp(['Eq.(11) N+2 original to new obj: ' num2str(obj_eq11_otn)]);
% disp(['Eq.(11) N+2 original to new eigen-gap: ' num2str(eg_eq11_otn)]);
% [obj_eq11_nto,eg_eq11_nto] = eq11_sdp_new_to_original(label,b_ind,n_sample,dA,dB,dL);
% disp(['Eq.(11) N+2 new to original obj: ' num2str(obj_eq11_nto)]);
% disp(['Eq.(11) N+2 new to original eigen-gap: ' num2str(eg_eq11_nto)]);

disp('Eq. closed form results====================================================================');
gamma=trace(cL)*1e2;
[y_new_eq_cf,current_obj_eq_cf,x_pred_eq_cf,err_count_eq_cf] = eq_closed_form_fast(gamma,label,n_sample,cL,b_ind);
disp(['Eq. closed-form obj: ' num2str(current_obj_eq_cf)]);
disp(['Eq.closed-form predicted labels: ' num2str(int8(x_pred_eq_cf'))]);
disp(['Eq. closed-form error_count: ' num2str(err_count_eq_cf)]);


% % % %% Eq. (12) min_{y,z} sum(y)+sum(b.*z) LP of the SDP-dual ==>
% % % %% Eq. (14) max_{w} sum(c.*w) LP standard form of the SDP-dual (interior-point)
% % % disp('Eq.(14) results====================================================================');
% % % [y_eq14,z_eq14,obj_eq14] = eq14(n_sample,...
% % %     b_ind,...
% % %     dL,...
% % %     db,...
% % %     rho,...
% % %     y_initial_scale,...
% % %     z_initial_scale,...
% % %     y_eq10,...
% % %     z_eq10);
% % % 
% % % disp(['Eq.(14) y: ' num2str(vec(y_eq14)')]);
% % % disp(['Eq.(14) z: ' num2str(vec(z_eq14)')]);
% % % disp(['Eq.(14) obj: ' num2str(obj_eq14)]);
% % % 
% % % %% Eq. (16) min_{v} sum(d.*v) LP-dual (SDP primal) (interior-point)
% % % disp('Eq.(16) results====================================================================');
% % % [v_eq16,obj_eq16] = eq16(n_sample,...
% % %     b_ind,...
% % %     dL,...
% % %     db,...
% % %     rho,...
% % %     y_initial_scale,...
% % %     z_initial_scale);
% % % 
% % % disp(['Eq.(16) v: ' num2str(vec(v_eq16)')]);
% % % disp(['Eq.(16) obj: ' num2str(obj_eq16)]);

results=[obj_eq5 error_count_eq5 ...
         obj_eq8 error_count_eq8...
         obj_eq10 error_count_eq10...
         current_obj_eq_cf err_count_eq_cf];





