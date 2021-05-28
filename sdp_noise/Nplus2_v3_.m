function [cvx_optval,x_pred,err_count,u_vec,alpha,eigen_gap,t_orig_end,dy,dz,new_H] = Nplus2_v3_(label,b_ind,n_sample,dA,dB,dL,u,alpha,sw)

db = 2*label(b_ind); % training labels x 2
dz_plus_idx = db < 0;
dz_minus_idx = db > 0;

AL = zeros(n_sample+1+1);
Bnp1 = zeros(n_sample+1+1);
Bnp2 = zeros(n_sample+1+1);

dz_ind_plus = b_ind(dz_plus_idx);
dz_ind_minus = b_ind(dz_minus_idx);

cL=-dL(1:n_sample,1:n_sample);
cL_offdia=cL;
cL_offdia(1:n_sample+1:end)=0;
t_orig=tic;


cvx_begin sdp
variables M(n_sample+2,n_sample+2);
minimize(sum(sum(ddL.*M)))
subject to
for i=1:n_sample
    M(i,i)==1;
end
M(n_sample+1,n_sample+1)==1;
M(n_sample+2,n_sample+2)==1;
for i=1:length(b_ind)
    sum(sum((dB(:,:,i).*M)))==2*label_b_ind(i);
end
for i=1:length(dz_ind_minus)
    Bnp1 = [dB(:,:,dz_ind_minus(i)) zeros(n_sample+1,1);zeros(n_sample+1,1)' 0];
    %dz(dz_ind_minus(i))<0;
    sum(sum((Bnp1.*M)))==db(dz_ind_minus(i));
end
for i=1:length(dz_ind_plus)
    dBnp2_i=dB(:,:,dz_ind_plus(i));
    dBnp2_i=dBnp2_i(:,end);
    Bnp2 = [zeros(n_sample+1) dBnp2_i;dBnp2_i' 0];
    %dz(dz_ind_plus(i))>0;
    sum(sum((Bnp2.*M)))==db(dz_ind_plus(i));
end
M>=0
cvx_end

cvx_begin sdp
variables dy(n_sample+1,1) dz(length(b_ind),1);
minimize(sum(dy)+db'*dz)
subject to
% for ii=1:n_sample
%     if ii<=length(b_ind)
%     dy(ii)+cL(ii,ii)==-sum(cL_offdia(ii,:))-dz(ii)+u(ii);
%     else
%     dy(ii)+cL(ii,ii)==-sum(cL_offdia(ii,:))+u(ii);    
%     end
% end
% dy(n_sample+1)==-sum(dz(dz_ind_minus))+u(n_sample+1)-alpha;
for i=1:n_sample+2
    if i~=n_sample+2
        if i==n_sample+1
            AL = AL + (sw*(dy(n_sample+1)+sum(dz))-sum(dz(dz_ind_minus))-alpha)...
                *[dA(:,:,i) zeros(n_sample+1,1); zeros(1,n_sample+1) 0];
        else
            AL = AL + (dy(i))*[dA(:,:,i) zeros(n_sample+1,1); zeros(1,n_sample+1) 0];
        end
    else
        dAnp2 = zeros(n_sample+2);
        dAnp2(n_sample+2,n_sample+2)=1;
        AL = AL + ((1-sw)*(dy(n_sample+1)+sum(dz))-sum(dz(dz_ind_plus))+alpha)...
            *dAnp2;
    end
end
for i=1:length(dz_ind_minus)
    Bnp1 = Bnp1 + dz(dz_ind_minus(i))*[dB(:,:,dz_ind_minus(i)) zeros(n_sample+1,1);zeros(n_sample+1,1)' 0];
    %dz(dz_ind_minus(i))<0;
end
for i=1:length(dz_ind_plus)
    dBnp2_i=dB(:,:,dz_ind_plus(i));
    dBnp2_i=dBnp2_i(:,end);
    Bnp2 = Bnp2 + dz(dz_ind_plus(i))*[zeros(n_sample+1) dBnp2_i;dBnp2_i' 0];
    %dz(dz_ind_plus(i))>0;
end
AL+Bnp1+Bnp2-[dL zeros(n_sample+1,1); zeros(1,n_sample+1) 0]>=0;
cvx_end
t_orig_end=toc(t_orig);
%% the solution new_H

new_H=AL+Bnp1+Bnp2-[dL zeros(n_sample+1,1); zeros(1,n_sample+1) 0];
new_H=(new_H+new_H')/2;

%% the original_H converted from new_H
original_H=-dL;

% original_H(1:n_sample+1+1:end)=diag(-dL)+[dy(1:n_sample); 2*dy(n_sample+1)+sum(dz(dz_ind_minus))-sum(dz(dz_ind_plus))];
original_H(1:n_sample+1+1:end)=diag(-dL)+dy;
original_H(b_ind,end)=dz;
original_H(end,b_ind)=dz;

original_H=(original_H+original_H')/2;

H_offdia=original_H;
H_offdia(1:n_sample+1+1:end)=0;

u_vec=diag(original_H)+sum(H_offdia,2);

%% eigen-gap
eigen_gap=min(eig(new_H))-min(eig(original_H));

%% first eigenvector of the converted original_H
% [v,~]=eig(original_H);
% fv_H=v(:,1);
rng(0);
fv_H_0=randn(n_sample+1,1);

[fv_H,~] = ...
    lobpcg_fv(...
    fv_H_0,...
    original_H,...
    1e-16,...
    1e3);

% fv_Hb_0=randn(n_sample+2,1);
% 
% [fv_Hb,~] = ...
%     lobpcg_fv(...
%     fv_Hb_0,...
%     new_H,...
%     1e-4,...
%     200);

% [v,~]=eig(new_H);
% fv_H=v(:,1);

db_plus_idx = db > 0;
db_minus_idx = db < 0;
db_plus_idx = b_ind(db_plus_idx);
db_minus_idx = b_ind(db_minus_idx);

w_positive=original_H(db_plus_idx,end); % corres. to db_plus_idx
w_positive=-w_positive;
w_negative=original_H(db_minus_idx,end); % corres. to db_minus_idx
w_negative=-w_negative;
delta_xNplus1_xi_positive=fv_H(end)-fv_H(db_plus_idx);
delta_xNplus1_xi_negative=fv_H(end)-fv_H(db_minus_idx);
sNplus1=sum(w_positive.*delta_xNplus1_xi_positive);
sNplus2=sum(w_negative.*delta_xNplus1_xi_negative);
alpha=(sNplus1-sNplus2)/(2*fv_H(end));

x_val = sign(label(b_ind(1))*sign(fv_H(1)))*sign(fv_H(1:n_sample));
% x_val = sign(fv_H(1:n_sample));
x_pred=sign(x_val);
err_count = sum(abs(sign(x_val) - label))/2;
end