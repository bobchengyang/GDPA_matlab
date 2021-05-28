% function [dy,dz,cvx_optval,db,x_pred,err_count,LA_obj] = eq10(label,b_ind,n_sample,dA,dB,dL,mu)
function [dy,dz,cvx_optval,db,x_pred,err_count] = eq10_beta(label,b_ind,n_sample,dA,dB,dL)
db = 2*label(b_ind); % training labels x 2
dM = zeros(n_sample+1,n_sample+1); % number of total samples + 1

beta=0.1;

cvx_begin sdp
variables dy(n_sample+1,1) dz(length(b_ind),1);
minimize(sum(dy) + db'*dz)
subject to
for i=1:n_sample
    dM = dM + dy(i)*dA(:,:,i);
end
dM=dM+beta^2*dy(n_sample+1)*dA(:,:,n_sample+1);
for i=1:length(b_ind)
    dM = dM + beta*dz(i)*dB(:,:,i);
end
dM-dL>=0
cvx_end

H=dM-dL;
H=(H+H')/2;

full_dM=full(dM);
full_dM = (full_dM+full_dM')/2;

fv_H_0=randn(n_sample+1,1);

[fv_H,lambda] = ...
    lobpcg_fv(...
    fv_H_0,...
    H,...
    1e-4,...
    200);

x_val = sign(label(b_ind(1))*sign(fv_H(1)))*sign(fv_H(1:n_sample));
% x_val = sign(fv_H(1:n_sample));
x_pred=sign(x_val(round(n_sample/2)+1:1:n_sample));
err_count = sum(abs(sign(x_val) - label))/2;
end

