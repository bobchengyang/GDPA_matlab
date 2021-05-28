function [dy,dz,cvx_optval,db] = eq10(label,b_ind,n_sample,dA,dB,dL)
db = 2*label(b_ind); % training labels x 2
dM = zeros(n_sample+1,n_sample+1); % number of total samples + 1
cvx_begin sdp
variables dy(n_sample+1,1) dz(length(b_ind),1);
minimize(sum(dy) + db'*dz)
subject to
for i=1:n_sample+1
    dM = dM + dy(i)*dA(:,:,i);
end
for i=1:length(b_ind)
    dM = dM + dz(i)*dB(:,:,i);
end
dM-dL>=0
cvx_end
end

