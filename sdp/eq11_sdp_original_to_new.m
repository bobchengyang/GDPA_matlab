function [cvx_optval,eigen_gap] = eq11_sdp_original_to_new(label,b_ind,n_sample,dA,dB,dL)

db = 2*label(b_ind); % training labels x 2
dM = zeros(n_sample+1,n_sample+1); % number of total samples + 1

beta=1e+6;

cvx_begin sdp
cvx_precision best;
variables dy(n_sample+1,1) dz(length(b_ind),1);
minimize(sum(dy(1:n_sample))+beta^2*dy(end) + db'*beta*dz)
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

original_H=dM-dL;
original_H=(original_H+original_H')/2;

u=dy(end)+sum(dz);

[new_H] = otn(label,b_ind,n_sample,dL,dy,dz,u);

scaling_D=eye(n_sample+1);
scaling_D(end,end)=1;
original_H_scaled=scaling_D*original_H*scaling_D;

dy_scaled=dy;
dy_scaled(end)=original_H_scaled(end,end);
dz_scaled=original_H_scaled(1:n_sample,end);
u_scaled=dy_scaled(end)+sum(dz_scaled);
[new_H_scaled] = otn(label,b_ind,n_sample,dL,dy_scaled,dz_scaled,u_scaled);

eigen_gap=min(eig(new_H))-min(eig(original_H));
eigen_gap_scaled=min(eig(new_H_scaled))-min(eig(original_H_scaled));
end