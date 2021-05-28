function [x_pred,err_count,cvx_optval] = eq8(dL,dB,n_sample,b_ind,label)
label_b_ind=label(b_ind);
cvx_begin sdp
variables M(n_sample+1,n_sample+1);
maximize(sum(sum(dL.*M)))
subject to
M(1:n_sample+1+1:end)==1;
for i=1:length(b_ind)
    sum(sum((dB(:,:,i).*M)))==2*label_b_ind(i);
end
M>=0
cvx_end

x_val=M(1:end-1,end);
x_pred=x_val(2:2:n_sample);
err_count = sum(abs(sign(x_val) - label))/2;
end
