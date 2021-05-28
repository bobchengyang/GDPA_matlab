function [cvx_optval,eigen_gap,u_vec,alpha] = eq11_sdp_iter(label,b_ind,n_sample,dA,dB,dL,u,alpha)

db = 2*label(b_ind); % training labels x 2
dz_plus_idx = db < 0;
dz_minus_idx = db > 0;

AL = zeros(n_sample+1+1);
Bnp1 = zeros(n_sample+1+1);
Bnp2 = zeros(n_sample+1+1);

dz_ind_plus = b_ind(dz_plus_idx);
dz_ind_minus = b_ind(dz_minus_idx);

cvx_begin sdp
variables dy(n_sample+1,1) dz(length(b_ind),1);
minimize(sum(dy)+db'*dz)
subject to
% for ii=1:n_sample
%     if ii<=length(b_ind)
%     dy(ii)==-dz(ii)+u(ii);
%     else
%     dy(ii)==u(ii);    
%     end
% end
dy(n_sample+1)==u(n_sample+1)-sum(dz(dz_ind_minus))-alpha;
for i=1:n_sample+2
    if i~=n_sample+2
        if i==n_sample+1
%             AL = AL + (0.5*dy(n_sample+1)+0.5*sum(dz(dz_ind_plus))-0.5*sum(dz(dz_ind_minus)))...
%                 *[dA(:,:,i) zeros(n_sample+1,1); zeros(1,n_sample+1) 0];
            AL = AL + (dy(i))...
                *[dA(:,:,i) zeros(n_sample+1,1); zeros(1,n_sample+1) 0];
        else
            AL = AL + (dy(i))*[dA(:,:,i) zeros(n_sample+1,1); zeros(1,n_sample+1) 0];
        end
    else
        dAnp2 = zeros(n_sample+2);
        dAnp2(n_sample+2,n_sample+2)=1;
%         AL = AL + (0.5*dy(n_sample+1)-0.5*sum(dz(dz_ind_plus))+0.5*sum(dz(dz_ind_minus)))...
%             *dAnp2;
        AL = AL + (u(n_sample+1)-sum(dz(dz_ind_plus))+alpha)...
            *dAnp2;
    end
end
for i=1:length(dz_ind_minus)
    Bnp1 = Bnp1 + dz(dz_ind_minus(i))*[dB(:,:,dz_ind_minus(i)) zeros(n_sample+1,1);zeros(n_sample+1,1)' 0];
    dz(dz_ind_minus(i))<0;
end
for i=1:length(dz_ind_plus)
    dBnp2_i=dB(:,:,dz_ind_plus(i));
    dBnp2_i=dBnp2_i(:,end);
    Bnp2 = Bnp2 + dz(dz_ind_plus(i))*[zeros(n_sample+1) dBnp2_i;dBnp2_i' 0];
    dz(dz_ind_plus(i))>0;
end
AL+Bnp1+Bnp2-[dL zeros(n_sample+1,1); zeros(1,n_sample+1) 0]>=0;
cvx_end

%% the solution new_H
new_H=AL+Bnp1+Bnp2-[dL zeros(n_sample+1,1); zeros(1,n_sample+1) 0];
new_H=(new_H+new_H')/2;

%% the original_H converted from new_H
original_H=-dL;

original_H(1:n_sample+1+1:end)=diag(-dL)+dy;
original_H(1:n_sample,end)=dz;
original_H(end,1:n_sample)=dz;

original_H=(original_H+original_H')/2;

H_offdia=original_H;
H_offdia(1:n_sample+1+1:end)=0;

u_vec=diag(original_H)+sum(H_offdia,2);

%% eigen-gap
eigen_gap=min(eig(new_H))-min(eig(original_H));

%% first eigenvector of the converted original_H
fv_H_0=randn(n_sample+1,1);

[fv_H,~] = ...
    lobpcg_fv(...
    fv_H_0,...
    original_H,...
    1e-4,...
    200);

%% self-loop adjustment weight alpha
db_plus_idx = db > 0;
db_minus_idx = db < 0;
db_plus_idx = b_ind(db_plus_idx);
db_minus_idx = b_ind(db_minus_idx);

w_positive=original_H(db_plus_idx,n_sample+1); % corres. to db_plus_idx
w_positive=-w_positive; % take the minus to get the weight
w_negative=original_H(db_minus_idx,n_sample+1); % corres. to db_minus_idx
w_negative=-w_negative; % take the minus to get the weight
delta_xNplus1_xi_positive=fv_H(end)-fv_H(db_plus_idx);
delta_xNplus1_xi_negative=fv_H(end)-fv_H(db_minus_idx);
sNplus1=sum(w_positive.*delta_xNplus1_xi_positive);
sNplus2=sum(w_negative.*delta_xNplus1_xi_negative);
alpha=(sNplus1-sNplus2)/(2*fv_H(end));
%u=dy(n_sample)+sum(dz);
end