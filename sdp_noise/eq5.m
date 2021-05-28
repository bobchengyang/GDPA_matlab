function [x_pred,err_count,cvx_optval] = eq5(cL,n_sample,b_ind,label)
cvx_begin sdp
variables X(n_sample,n_sample) x(n_sample,1);
minimize(trace(cL*X))
subject to
[X x; x' 1]>=0; 
X(1:n_sample+1:end) == 1; 
x(b_ind) == label(b_ind)
cvx_end

% F = [M>=0, X(dia_idx) == ones(n_sample,1), x(b_ind) == label(b_ind)];
% F = [M>=0, X(dia_idx) == ones(n_sample,1), x(b_ind) == label(b_ind), X(m_ind) >= -ones(1,n_sample^2), X(m_ind) <= ones(1,n_sample^2)];
%optimize(F,t,sdpsettings('solver','penlab'));
% optimize(F,t,sdpsettings('solver','sedumi'));

% t_val = value(t);
% X_val = value(X);
% x_val = value(x);
x_val=x;
x_pred=sign(x_val);
err_count = sum(abs(sign(x_val) - label))/2;
end

