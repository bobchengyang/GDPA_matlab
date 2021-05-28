function [dy,dz,cvx_optval,db,x_pred,err_count] = eq11_sdp(label,b_ind,n_sample,dA,dB,dL)

alpha = 0.01;

db = 2*label(b_ind); % training labels x 2
dz_plus_idx = db < 0;
dz_minus_idx = db > 0;
AL_plus = zeros(n_sample+1); % number of total samples + 1
AL_minus = zeros(n_sample+1);

B_plus = zeros(n_sample+1);
B_minus = zeros(n_sample+1);

b_ind_plus = b_ind(dz_plus_idx);
b_ind_minus = b_ind(dz_minus_idx);

cvx_begin sdp
variables dy(n_sample+1,1) dz(length(b_ind_plus),1);
minimize(sum(dy) + db(b_ind_plus)'*dz)
subject to
for i=1:n_sample+1
    AL_plus = AL_plus + dy(i)*dA(:,:,i);
end
AL_plus = AL_plus - dL;
for i=1:length(b_ind_plus)
    B_plus = B_plus + dz(i)*dB(:,:,b_ind_plus(i));
    dz(i)>0;
end
alpha*AL_plus+B_plus>=0;
cvx_end

cvx_plus=cvx_optval;

cvx_begin sdp
variables dy(n_sample+1,1) dz(length(b_ind_minus),1);
minimize(sum(dy) + db(b_ind_minus)'*dz)
subject to
for i=1:n_sample+1
    AL_minus = AL_minus + dy(i)*dA(:,:,i);
end
AL_minus = AL_minus - dL;
for i=1:length(b_ind_minus)
    B_minus = B_minus + dz(i)*dB(:,:,b_ind_minus(i));
    dz(i)<0;
end
(1-alpha)*AL_minus+B_minus>=0;
cvx_end

min(eig(alpha*AL_plus+B_plus+(1-alpha)*AL_minus+B_minus))

cvx_minus=cvx_optval;

cvx_net=cvx_plus+cvx_minus;


% % % db = 2*label(b_ind); % training labels x 2
% % % g_0 = -1*db; % previously computed g
% % % yn1 = 1; % set initial y_{N+1} as 1
% % % db_tilde = db./g_0; % db_tilde
% % % db_tilde = [db_tilde;zeros(3,1)];
% % % 
% % % tol_set=1e-5;
% % % tol=Inf;
% % % iter=0;
% % % while tol>tol_set
% % %     iter=iter+1;
% % %     %% solve {y}_{i=1}^N and G while y_{N+1} is fixed
% % %     cvx_begin sdp
% % %     variables dy(n_sample,1) G(n_sample,n_sample);
% % %     minimize(sum(dy) + db_tilde'*diag(G))
% % %     subject to
% % %     [diag(dy)+cL-G/yn1 zeros(n_sample); zeros(n_sample) G]>=0;
% % %     cvx_end   
% % %     
% % %     %% update g_0
% % %     fv_g_0=randn(n_sample,1);
% % %     
% % %     [fv_g,~] = ...
% % %         lobpcg_fv(...
% % %         fv_g_0,...
% % %         G,...
% % %         1e-4,...
% % %         200);
% % %     g_0 = fv_g;
% % %     db_tilde = db./g_0; % db_tilde
% % %     
% % %    %% update y_{N+1} while {y}_{i=1}^N and G are fixed
% % %     yn1=(g_0'/(diag(dy)+cL))*g_0;     
% % %     
% % %     if iter==1
% % %         tol=norm(cvx_optval);
% % %     else
% % %         tol=norm(tol-cvx_optval);
% % %     end
% % % end

end

