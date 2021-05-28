function [cvx_optval,eigen_gap,u_vec,alpha] = eq11_sdp_original_to_new_Nplus2(label,b_ind,n_sample,dA,dB,dL)

db = 2*label(b_ind); % training labels x 2
dM = zeros(n_sample+1,n_sample+1); % number of total samples + 1

cvx_begin sdp
cvx_precision best;
variables dy(n_sample+1,1) dz(length(b_ind),1);
minimize(sum(dy) + db'*dz)
subject to
for i=1:n_sample
    dM = dM + dy(i)*dA(:,:,i);
end
dM=dM+dy(n_sample+1)*dA(:,:,n_sample+1);
for i=1:length(b_ind)
    dM = dM + dz(i)*dB(:,:,i);
end
dM-dL>=0
cvx_end

H=dM-dL;
H=(H+H')/2;

H_offdia=H;
H_offdia(1:n_sample+1+1:end)=0;

u_vec=diag(H)+sum(H_offdia,2);

fv_H_0=randn(n_sample+1,1);

[fv_H,~] = ...
    lobpcg_fv(...
    fv_H_0,...
    H,...
    1e-4,...
    200);

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

original_H=dM-dL;
original_H=(original_H+original_H')/2;

u=H(end,end)+sum(dz);

[new_H] = otn(label,b_ind,n_sample,dL,dy,dz,u,alpha);

eigen_gap=min(eig(new_H))-min(eig(original_H));
end