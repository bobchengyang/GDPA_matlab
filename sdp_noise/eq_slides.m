function [dy] = eq_slides(cL,n_sample,b_ind,label)
cvx_begin sdp
variables dy(n_sample,1);
minimize(sum(dy)+)
subject to
diag(dy)-cL>=0
cvx_end
cvx_obj1=cvx_optval;

cvx_begin sdp
variables X(n_sample,n_sample);
minimize(trace(cL*X))
subject to
X(1:n_sample+1:end)==1;
X>=0;
X(1,b_ind(2:end)) == label(b_ind(2:end))'.*label(b_ind(1));
X(b_ind(2:end),1) == label(b_ind(2:end)).*label(b_ind(1));
cvx_end
cvx_obj2=cvx_optval;
end

