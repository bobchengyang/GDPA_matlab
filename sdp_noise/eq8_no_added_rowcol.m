function [x_pred,err_count,cvx_optval] = eq8_no_added_rowcol(dL,dB,n_sample,b_ind,label)
label_b_ind=label(b_ind);
cvx_begin sdp
variables M(n_sample,n_sample);
maximize(sum(sum(dL.*M)))
subject to
M(1:n_sample+1:end)==1;
for i=1:length(b_ind)
    if b_ind(i)==1
    %sum(sum((dB(:,:,i).*M)))==2;    
    else
    sum(sum((dB(:,:,i).*M)))==2*label_b_ind(i)*label_b_ind(1);
    end
end
M>=0
cvx_end

fv_M_0=randn(n_sample,1);

[fv_M,~] = ...
    lobpcg_fv(...
    fv_M_0,...
    -M,...
    1e-4,...
    200);

% x_val=fv_M;
x_val = label(b_ind(1))*M(:,1);        %% GC: retrieve labels from 1st column
x_pred=sign(x_val(round(n_sample/2)+1:1:n_sample));
err_count = sum(abs(sign(x_val) - label))/2;
end

