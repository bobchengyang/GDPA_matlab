function [x_pred,err_count,cvx_optval,t_orig_end] = eq8(dL,dB,n_sample,b_ind,label,label_noisy)
label_b_ind=label_noisy(b_ind);

t_orig=tic;
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
t_orig_end=toc(t_orig);

x_val=M(1:end-1,end);
x_pred=sign(x_val);
% err_count = sum(abs(sign(x_val) - label))/2;
err_count = sum(abs(sign(x_pred(length(b_ind)+1:end)) - label(length(b_ind)+1:end)))/2;
end