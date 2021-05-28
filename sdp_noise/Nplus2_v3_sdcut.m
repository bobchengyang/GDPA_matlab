function [obj,x_pred,err_count,u_vec,alpha,eigen_gap,t_orig_end,dy,dz,H] = Nplus2_v3_sdcut(label,label_noisy,b_ind,n_sample,dA,dB,dL,u,alpha,sw)

db = 2*label_noisy(b_ind); % training labels x 2
dz_plus_idx = db < 0;
dz_minus_idx = db > 0;
dz_ind_plus = b_ind(dz_plus_idx);
dz_ind_minus = b_ind(dz_minus_idx);

%% this is for CDCS ADMM SDP==========================================
% % % % % % % opts.maxIter = 1e+4;
% % % % % % % opts.relTol  = 1e-3;
% % % % % % % opts.solver = 'dual';
% % % % % % % opts.verbose = 0;
A=zeros(n_sample+2+length(b_ind),(n_sample+2)^2);
for i=1:n_sample
    A(i,:)=vec([dA(:,:,i) zeros(n_sample+1,1); zeros(1,n_sample+1) 0])';
end

Bnp1 = zeros(n_sample+2);
Bnp2 = zeros(n_sample+2);
for i=1:length(dz_ind_minus)
    Bnp1 = Bnp1+[dB(:,:,dz_ind_minus(i)) zeros(n_sample+1,1);zeros(n_sample+1,1)' 0];
    %dz(dz_ind_minus(i))<0;
end
for i=1:length(dz_ind_plus)
    dBnp2_i=dB(:,:,dz_ind_plus(i));
    dBnp2_i=dBnp2_i(:,end);
    Bnp2 = Bnp2+[zeros(n_sample+1) dBnp2_i;dBnp2_i' 0];
    %dz(dz_ind_plus(i))>0;
end
dA_nplus1=[dA(:,:,n_sample+1) zeros(n_sample+1,1); zeros(1,n_sample+1) 0];
A(n_sample+1,:)=vec(sw*(dA_nplus1+Bnp1+Bnp2)-Bnp1)';
dA_nplus2=zeros(n_sample+2);
dA_nplus2(n_sample+2,n_sample+2)=1;
A(n_sample+2,:)=vec((1-sw)*(dA_nplus2+Bnp1+Bnp2)-Bnp2)';

for i=1:length(dz_ind_minus)
    Bnp1 = [dB(:,:,dz_ind_minus(i)) zeros(n_sample+1,1);zeros(n_sample+1,1)' 0];
    %dz(dz_ind_minus(i))<0;
    A(n_sample+2+dz_ind_minus(i),:)=vec(Bnp1)';
end
for i=1:length(dz_ind_plus)
    dBnp2_i=dB(:,:,dz_ind_plus(i));
    dBnp2_i=dBnp2_i(:,end);
    Bnp2 = [zeros(n_sample+1) dBnp2_i;dBnp2_i' 0];
    %dz(dz_ind_plus(i))>0;
    A(n_sample+2+dz_ind_plus(i),:)=vec(Bnp2)';
end

b=zeros(n_sample+2+length(b_ind),1);
b(1:n_sample)=1;

b(n_sample+1)=sw*(1+sum(db))-sum(db(dz_ind_minus)); %?
b(n_sample+2)=(1-sw)*(1+sum(db))-sum(db(dz_ind_plus)); %?

b(n_sample+2+1:n_sample+2+length(b_ind))=db;

dL=dL/alpha;
dL(n_sample+1,n_sample+1)=1;
ddL=[-dL zeros(n_sample+1,1);zeros(n_sample+1,1)' 1];
% % % % % % % c=-vec([-dL zeros(n_sample+1,1);zeros(n_sample+1,1)' alpha])';
% % % % % % % 
% % % % % % % At=A';
% % % % % % % K.s = n_sample+2; % PSD matrix dimension
% % % % % % % [x_cdcs,y_cdcs,z_cdcs,info_cdcs] = cdcs(At,b,c,K,opts); % SDP via an ADMM approach.
% % % % % % % 
% % % % % % % H=reshape(z_cdcs,[n_sample+2 n_sample+2]);
% % % % % % % y_cdcs=-y_cdcs;
% % % % % % % dy=[y_cdcs(1:n_sample); y_cdcs(n_sample+1)+y_cdcs(n_sample+2)];
% % % % % % % dz=y_cdcs(n_sample+2+1:n_sample+2+length(b_ind));
% % % % % % % obj=info_cdcs.cost;
% % % % % % % t_orig_end=info_cdcs.time.admm;
%% this is for CDCS ADMM SDP ends=====================================

%% this is for SDCUT SDP==============================================
% get data and options
data.A=ddL; % mat(n_sample+1,n_sample+1)
B=cell(1,n_sample+2+length(b_ind));
for i=1:n_sample+2+length(b_ind)
    B{i}=reshape(A(i,:),[n_sample+2 n_sample+2]);
end
data.B=B; % cell(1,n_sample+1+length(b_ind))
data.b=b; % vec(n_sample+1+length(b_ind),1)
data.u_init=zeros(n_sample+2+length(b_ind),1);
data.u_init(b_ind)=1;
data.u_init(n_sample+1)=length(dz_ind_minus);
data.u_init(n_sample+2)=length(dz_ind_plus);
data.u_init(n_sample+2+dz_ind_minus)=-1;
data.u_init(n_sample+2+dz_ind_plus)=1;
data.l_bbox=zeros(n_sample+2+length(b_ind),1)-Inf; 
data.u_bbox=zeros(n_sample+2+length(b_ind),1)+Inf;

% % % dz_plus_idx = db < 0;
% % % dz_minus_idx = db > 0;
% % % dz_ind_plus = b_ind(dz_plus_idx);
% % % dz_ind_minus = b_ind(dz_minus_idx);
% % % data.l_bbox(n_sample+1+dz_ind_plus)=0;
% % % data.u_bbox(n_sample+1+dz_ind_minus)=0;

opts.sigma=1e-5; % between 1e-4 and 1e-2 according to the cvpr'13 paper
opts.lbfgsb_maxIts=1e2;
opts.lbfgsb_factr=1e7;
opts.lbfgsb_pgtol=1e-3;
opts.lbfgsb_m=2e2;
opts.lbfgsb_printEvery=1e2;
% solve dual
[u_opt,C_minus_opt,results]=eq_10_solve_dual_lbfgsb(data,opts);
% get optimal lifted primal X=x*x'
X_opt = (-0.5 / opts.sigma) * C_minus_opt;
% recover x from X=x*x'
dy=[u_opt(1:n_sample); u_opt(n_sample+1)+u_opt(n_sample+2)];
dz=u_opt(n_sample+2+1:n_sample+2+length(b_ind));
obj=results.obj;
t_orig_end=results.time;
disp(['sdcut lbfgsb iter: ' num2str(results.iters)]);
% H=[-dL(1:n_sample,1:n_sample)+diag(dy(1:n_sample)) [dz;zeros(n_sample-length(b_ind),1)];...
%     [dz;zeros(n_sample-length(b_ind),1)]' dy(n_sample+1)];
cL=-dL(1:n_sample,1:n_sample);
alpha=1;
[H] = construct_H(sw,n_sample,...
    cL,...
    u,...
    alpha,...
    dy,...
    dz,...
    dz_ind_plus,...
    dz_ind_minus,...
    3);
%% this is for SDCUT SDP ends=========================================

%% the original_H converted from new_H
dL(n_sample+1,n_sample+1)=0;
original_H=-dL;

% original_H(1:n_sample+1+1:end)=diag(-dL)+[dy(1:n_sample); 2*dy(n_sample+1)+sum(dz(dz_ind_minus))-sum(dz(dz_ind_plus))];
original_H(1:n_sample+1+1:end)=diag(-dL)+dy;
original_H(b_ind,end)=dz;
original_H(end,b_ind)=dz;
original_H=(original_H+original_H')/2;

rng('default');
rng(0);
fv_H_0=randn(n_sample+1,1);

[fv_H,~] = ...
    lobpcg_fv(...
    fv_H_0,...
    original_H,...
    1e-16,...
    1e3);

x_val = sign(label_noisy(b_ind(1))*sign(fv_H(1)))*sign(fv_H(1:n_sample));
x_pred=sign(x_val);
% err_count = sum(abs(sign(x_val) - label))/2;
err_count = sum(abs(sign(x_pred(length(b_ind)+1:end)) - label(length(b_ind)+1:end)))/2;
x_val2 = sign(X_opt(1:n_sample,n_sample+1));
% x_pred=sign(x_val);
% err_count2 = sum(abs(sign(x_val2) - label))/2;
err_count2 = sum(abs(sign(x_val2(length(b_ind)+1:end)) - label(length(b_ind)+1:end)))/2;
disp(['error_count dual: ' num2str(err_count) ' | error_count primal: ' num2str(err_count2)]);

H_offdia=H;
H_offdia(1:n_sample+1+1:end)=0;

u_vec=diag(H)+sum(H_offdia,2);

%% eigen-gap
eigen_gap=min(eig(H))-min(eig(original_H));

db_plus_idx = db > 0;
db_minus_idx = db < 0;
db_plus_idx = b_ind(db_plus_idx);
db_minus_idx = b_ind(db_minus_idx);

w_positive=H(db_plus_idx,end); % corres. to db_plus_idx
w_positive=-w_positive;
w_negative=H(db_minus_idx,end); % corres. to db_minus_idx
w_negative=-w_negative;
delta_xNplus1_xi_positive=fv_H(end)-fv_H(db_plus_idx);
delta_xNplus1_xi_negative=fv_H(end)-fv_H(db_minus_idx);
sNplus1=sum(w_positive.*delta_xNplus1_xi_positive);
sNplus2=sum(w_negative.*delta_xNplus1_xi_negative);
alpha=(sNplus1-sNplus2)/(2*fv_H(end));
%% =======================
end