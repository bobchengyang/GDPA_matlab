function [x_pred,err_count,cvx_optval,t_orig_end] = eq8_no_label(dL,n_sample,b_ind,label)
label_b_ind=label(b_ind);

t_orig=tic;

cvx_begin sdp
variables M(n_sample+1,n_sample+1);
maximize(sum(sum(dL.*M)))
subject to
M(1:n_sample+1+1:end)==1;
M>=0
cvx_end

t_orig_end=toc(t_orig);

x_val=M(1:end-1,end);
x_pred=sign(x_val);
err_count = sum(abs(sign(x_val) - label))/2;
end

