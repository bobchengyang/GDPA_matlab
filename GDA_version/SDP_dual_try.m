clear;
clc;
close all;

%%=======change the following things for experiments======
n_sample=12; % total number of samples to be experimented
rho=1e-12; % PSD parameter
y_initial_scale=1e1; % for y initialization (LP variables in Eq. (12))
z_initial_scale=1e0; % for z initialization (LP variables in Eq. (12)) 
%%========================================================

[cL,label] = dataLoader_normalized('monk1.csv',n_sample);
b_ind = 1:2:n_sample; % every other sample / ~50% training sample
disp(['True labels: ' num2str(label(2:2:n_sample)')]);

%% Eq. (5) min_{x,X} tr(cL*X) SDP (interior-point)
disp('Eq.(5) results====================================================================');
[x_pred_eq5,error_count_eq5,obj_eq5]=eq5(cL,n_sample,b_ind,label);
disp(['Eq.(5) predicted labels: ' num2str(x_pred_eq5')]);
disp(['Eq.(5) error_count: ' num2str(error_count_eq5)]);
disp(['Eq.(5) obj: ' num2str(obj_eq5)]);

%% Eq. (8) max_{M} sum(vec(L.*M)) SDP standard form (interior-point) 
disp('Eq.(8) results====================================================================');
dL = [-cL zeros(n_sample,1); zeros(1,n_sample) 0]; % eq. (6)
ei_l = eye(n_sample+1);   %% two identity matrices
ei_s = eye(n_sample);
for i=1:length(b_ind)
    dB(:,:,i) = [zeros(n_sample,n_sample) ei_s(:,b_ind(i)); ei_s(b_ind(i),:) 0]; % eq. (7), where dB is a stacked matrix
end
[x_pred_eq8,error_count_eq8,obj_eq8] = eq8(dL,dB,n_sample,b_ind,label);

disp(['Eq.(8) predicted labels: ' num2str(x_pred_eq8')]);
disp(['Eq.(8) error_count: ' num2str(error_count_eq8)]);
disp(['Eq.(8) obj: ' num2str(obj_eq8)]);

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


%% Eq. (10) min_{y,z} sum(y)+sum(b.*z) SDP-dual (interior-point)   
disp('Eq.(10) results====================================================================');
for i=1:n_sample+1
    dA(:,:,i) = diag(ei_l(:,i)); % eq. (7), where dA is a stacked matrix where each of them has only 1 non-zero entry
end
[y_eq10,z_eq10,obj_eq10,db] = eq10(label,b_ind,n_sample,dA,dB,dL);

disp(['Eq.(10) y: ' num2str(vec(y_eq10)')]);
disp(['Eq.(10) z: ' num2str(vec(z_eq10)')]);
disp(['Eq.(10) obj: ' num2str(obj_eq10)]);

%% Eq. (12) min_{y,z} sum(y)+sum(b.*z) LP of the SDP-dual ==>
%% Eq. (14) max_{w} sum(c.*w) LP standard form of the SDP-dual (interior-point)
disp('Eq.(14) results====================================================================');
[y_eq14,z_eq14,obj_eq14] = eq14(n_sample,...
    b_ind,...
    dL,...
    db,...
    rho,...
    y_initial_scale,...
    z_initial_scale);

disp(['Eq.(14) y: ' num2str(vec(y_eq14)')]);
disp(['Eq.(14) z: ' num2str(vec(z_eq14)')]);
disp(['Eq.(14) obj: ' num2str(obj_eq14)]);

%% Eq. (16) min_{v} sum(d.*v) LP-dual (SDP primal) (interior-point)
disp('Eq.(16) results====================================================================');
[v_eq16,obj_eq16] = eq16(n_sample,...
    b_ind,...
    dL,...
    db,...
    rho,...
    y_initial_scale,...
    z_initial_scale);

disp(['Eq.(16) v: ' num2str(vec(v_eq16)')]);
disp(['Eq.(16) obj: ' num2str(obj_eq16)]);





