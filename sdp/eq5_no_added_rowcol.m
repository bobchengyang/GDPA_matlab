function [x_pred,err_count,cvx_optval] = eq5_no_added_rowcol(cL,n_sample,b_ind,label)
cvx_begin sdp
variables X(n_sample,n_sample);
minimize(trace(cL*X))
subject to
X>=0; 
X(1:n_sample+1:end) == 1; 
% X(1:length((b_ind)),1:length((b_ind)))==label(b_ind)*label(b_ind)';
X(1,b_ind(2:end)) == label(b_ind(2:end))'.*label(b_ind(1));
X(b_ind(2:end),1) == label(b_ind(2:end)).*label(b_ind(1));
cvx_end

% F = [M>=0, X(dia_idx) == ones(n_sample,1), x(b_ind) == label(b_ind)];
% F = [M>=0, X(dia_idx) == ones(n_sample,1), x(b_ind) == label(b_ind), X(m_ind) >= -ones(1,n_sample^2), X(m_ind) <= ones(1,n_sample^2)];
%optimize(F,t,sdpsettings('solver','penlab'));
% optimize(F,t,sdpsettings('solver','sedumi'));

% t_val = value(t);
% X_val = value(X);
% x_val = value(x);

X = (X+X')/2;

fv_X_0=randn(n_sample,1);

[fv_X,~] = ...
    lobpcg_fv(...
    fv_X_0,...
    -X,...
    1e-4,...
    200);

%x_val=fv_X;
x_val = label(1)*X(:,1);        %% GC: retrieve labels from 1st column
x_pred=sign(x_val(round(n_sample/2)+1:1:n_sample));
err_count = sum(abs(sign(x_val) - label))/2;
end

