function [current_obj,x_pred,err_count,u_vec,alpha,eigen_gap,t_orig_end] = ...
    Nplus2_self_loop_no_iter_cheat_GDPA(label,b_ind,n_sample,cL,u,alpha,sw,...
    dy_LP_test_init,dz_LP_test_init,new_H_LP_test_init,...
    rho)

%% LP settings
% options=optimoptions('linprog','Algorithm','interior-point','display','none'); % linear program (LP) setting for Frank-Wolfe algorithm
options=optimoptions('linprog','Algorithm','interior-point'); % linear program (LP) setting for Frank-Wolfe algorithm
options.OptimalityTolerance=1e-5; % LP optimality tolerance
options.ConstraintTolerance=1e-5; % LP interior-point constraint tolerance

db = 2*label(b_ind); % training labels x 2
dz_plus_idx = db < 0;
dz_minus_idx = db > 0;

dz_ind_plus = b_ind(dz_plus_idx);
dz_ind_minus = b_ind(dz_minus_idx);

%% initialize a psd matrix ABL
% u(n_sample+1)=1e1;
% alpha=1e1;
% % % rng(0);
% dy_LP_test_init=[ones(n_sample+1,1);sum(db(db>0))/2;-sum(db(db<0))/2];
% dz_LP_test_init=-db*1;

scalee=1/alpha;
dy_LP_test_init=dy_LP_test_init*scalee;
dz_LP_test_init=dz_LP_test_init*scalee;
cL=cL*scalee;
alpha=1;

[initial_H] = construct_H(sw,n_sample,...
    cL,...
    u,...
    alpha,...
    dy_LP_test_init,...
    dz_LP_test_init,...
    dz_ind_plus,...
    dz_ind_minus,...
    3);

diff_H=norm(vec(new_H_LP_test_init)-vec(initial_H));
disp(['diff H: ' num2str(diff_H)]);
rng(0);
fv_H=randn(n_sample+2,1);

initial_obj=sum(dy_LP_test_init)+db'*dz_LP_test_init;
disp(['v3 LP main iteration ' num2str(0) ' | current obj: ' num2str(initial_obj) ' | mineig: ' num2str(min(eig(initial_H)))]);

t_orig=tic;

tol_set=1e-5;
tol=Inf;
loop_i=0;
while tol>tol_set
    if loop_i==0
        [fv_H,...
            scaled_M,...
            scaled_factors] = ...
            compute_scalars(...
            initial_H,...
            fv_H); % compute scalars
        
        scaled_M_offdia=scaled_M;
        scaled_M_offdia(1:n_sample+2+1:end)=0;
        leftEnds=diag(initial_H)-sum(abs(scaled_M_offdia),2);
        leftEnds_diff=sum(abs(leftEnds-mean(leftEnds)));
        disp(['v3 LP before LP LeftEnds mean: ' num2str(mean(leftEnds)) ' | LeftEnds difference: ' num2str(leftEnds_diff)]);
    end
    
    [y,z,current_obj]=LP_core_Nplus2( ...
        n_sample,...
        scaled_M,...
        scaled_factors,...
        rho,...
        db,...
        options,...
        cL,...
        b_ind,...
        u,...
        alpha,...
        sw,...
        dz_ind_plus,...
        dz_ind_minus);
    
    [updated_H] = construct_H(sw,n_sample,...
    cL,...
    u,...
    alpha,...
    y,...
    z,...
    dz_ind_plus,...
    dz_ind_minus,...
    3);

% %% the solution new_H
% updated_H=(updated_H+updated_H')/2;
% 
% %% the original_H converted from new_H
% original_H=[cL zeros(n_sample,1);zeros(n_sample,1)' 0];
% % original_H(1:n_sample+1+1:end)=[diag(cL);0]+[y;-sum(z(dz_ind_minus))+0.5*u(n_sample+1)-alpha];
% original_H(1:n_sample+1+1:end)=[diag(cL);0]+y;
% % original_H(1:n_sample+1+1:end)=original_H(1:n_sample+1+1:end)+[y(1:n_sample); sum(y(n_sample+1)-sum(z(dz_ind_plus)))]';
% original_H(b_ind,n_sample+1)=z;
% original_H(n_sample+1,b_ind)=z;
% 
% original_H=(original_H+original_H')/2;
% 
% rng(0);
% fv_Ho_0=randn(n_sample+1,1);
% [alpha] = compute_alpha(...
%     n_sample,...
%     db,...
%     b_ind,...
%     original_H,...
%     fv_Ho_0);

    [fv_H,...
        scaled_M,...
        scaled_factors] = ...
        compute_scalars(...
        updated_H,...
        fv_H); % compute scalars
    
    scaled_M_offdia=scaled_M;
    scaled_M_offdia(1:n_sample+2+1:end)=0;
    leftEnds=diag(updated_H)-sum(abs(scaled_M_offdia),2);
    leftEnds_diff=sum(abs(leftEnds-mean(leftEnds)));
    disp(['v3 LP after LP LeftEnds mean: ' num2str(mean(leftEnds)) ' | LeftEnds difference: ' num2str(leftEnds_diff)]);
    
    disp(['v3 LP obj ' num2str(current_obj)]);
    tol=norm(current_obj-initial_obj);
    initial_obj=current_obj;
    loop_i=loop_i+1;
end

t_orig_end=toc(t_orig);

%% the solution new_H
updated_H=(updated_H+updated_H')/2;

%% the original_H converted from new_H
original_H=[cL zeros(n_sample,1);zeros(n_sample,1)' 0];
% original_H(1:n_sample+1+1:end)=[diag(cL);0]+[y;-sum(z(dz_ind_minus))+0.5*u(n_sample+1)-alpha];
original_H(1:n_sample+1+1:end)=[diag(cL);0]+y;
% original_H(1:n_sample+1+1:end)=original_H(1:n_sample+1+1:end)+[y(1:n_sample); sum(y(n_sample+1)-sum(z(dz_ind_plus)))]';
original_H(b_ind,n_sample+1)=z;
original_H(n_sample+1,b_ind)=z;

original_H=(original_H+original_H')/2;

H_offdia=original_H;
H_offdia(1:n_sample+1+1:end)=0;

u_vec=diag(original_H)+sum(H_offdia,2);

%% eigen-gap
eigen_gap=min(eig(updated_H))-min(eig(original_H));

% test_v=-1e4-200;
% rr=zeros(100,2);
% for ii=1:100
%     test_v=test_v+200;
%     original_H(end,end)=test_v;
%% first eigenvector of the converted original_H
rng(0);
% fv_H_0=randn(n_sample+1,1);
% 
% [fv_H,~] = ...
%     lobpcg_fv(...
%     fv_H_0,...
%     original_H,...
%     1e-12,...
%     20000);
[v,~]=eig(original_H);
fv_H=v(:,1);

[alpha] = compute_alpha(...
    n_sample,...
    db,...
    b_ind,...
    original_H,...
    fv_H);

x_val = sign(label(b_ind(1))*sign(fv_H(1)))*sign(fv_H(1:n_sample));
% x_val = sign(fv_H(1:n_sample));
x_pred=sign(x_val);
err_count = sum(abs(sign(x_val) - label))/2;
% rr(ii,1)=test_v;
% rr(ii,2)=err_count;
% 
% end
% 
% figure();plot(rr(:,1),rr(:,2));
end