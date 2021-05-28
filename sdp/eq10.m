function [dy,dz,cvx_optval,db,x_pred,err_count,t_orig_end,u_vec,alpha] = eq10(label,b_ind,n_sample,dA,dB,dL)
db = 2*label(b_ind); % training labels x 2
dM = zeros(n_sample+1,n_sample+1); % number of total samples + 1

t_orig=tic;
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
% dM>=0
cvx_end
t_orig_end=toc(t_orig);

H=dM-dL;
H=(H+H')/2;

H_offdia=H;
H_offdia(1:n_sample+1+1:end)=0;

u_vec=diag(H)+sum(H_offdia,2);

fv_H_0=randn(n_sample+1,1);

[fv_H,lambda] = ...
    lobpcg_fv(...
    fv_H_0,...
    H,...
    1e-16,...
    1e3);

x_val = sign(label(b_ind(1))*sign(fv_H(1)))*sign(fv_H(1:n_sample));
% x_val = sign(fv_H(1:n_sample));
x_pred=sign(x_val);
err_count = sum(abs(sign(x_val) - label))/2;
disp(['min eig H: ' num2str(lambda)]);
disp(['energy in b_ind n_sample+1: ' num2str(norm(fv_H([b_ind n_sample+1])))...
    ' | energy in z: ' num2str(norm(fv_H(length(b_ind)+1:end-1)))]);

db_plus_idx = db > 0;
db_minus_idx = db < 0;
db_plus_idx = b_ind(db_plus_idx);
db_minus_idx = b_ind(db_minus_idx);

w_positive=H(db_plus_idx,end); % corres. to db_plus_idx
w_positive=-w_positive;
w_negative=H(db_minus_idx,end); % corres. to db_minus_idx
w_negative=-w_negative;
delta_xNplus1_xi_positive=fv_H(end)-fv_H(db_plus_idx);
delta_xNplus1_xi_negative=fv_H(end)-fv_H(db_minus_idx);
sNplus1=sum(w_positive.*delta_xNplus1_xi_positive);
sNplus2=sum(w_negative.*delta_xNplus1_xi_negative);
alpha=(sNplus1-sNplus2)/(2*fv_H(end));

% H_tilde=H(2:end,2:end);
% gamma=H_tilde-(1/H(1,1))*H(2:end,1)*H(2:end,1)';
% gamma=(gamma+gamma')/2;
% gamma_0=BFS_Balanced(-gamma);
% if isequal(gamma_0,-gamma)
%     disp('gamma balanced.');
% else
%     disp('gamma NOT balanced.');
% end

% % % % %% check graph balanceness of H_tilde
% % % % 
% % % % dz_full=zeros(n_sample,1);
% % % % dz_full(b_ind)=dz;
% % % % dz=dz_full;
% % % % 
% % % % H=dM-dL;
% % % % H_tilde=H(1:end-1,1:end-1);
% % % % gamma=H_tilde-(1/dy(end))*dz*dz';
% % % % gamma=(gamma+gamma')/2;
% % % % gamma_0=BFS_Balanced(-gamma);
% % % % if isequal(gamma_0,-gamma)
% % % %     disp('gamma balanced.');
% % % % else
% % % %     disp('gamma NOT balanced.');
% % % % end
% % % % % sign(dz*dz')+sign(gamma)
% % % % 
% % % % %% try GSP label assignment after the above SDP dual
% % % % known_idx=logical(zeros(n_sample,1));
% % % % known_idx(b_ind)=1;
% % % % n_known=length(find(known_idx));
% % % % H=zeros(n_known,n_sample);
% % % % for H_i=1:n_known
% % % %     H(H_i,b_ind(H_i))=1;
% % % % end
% % % % % cvx_begin
% % % % % variable x(n_sample,1)
% % % % % minimize(norm(label(b_ind)-H*x,2)+x'*gamma*x)
% % % % % cvx_end
% % % % x_closed=(H'*H+mu*gamma)^(-1)*H'*label(b_ind); % see Yuanchao's TSP'2020 graph sampling Eq. (7) and Eq. (8) x*=(H'*H+\mu*L)^(-1)H'*y
% % % % LA_obj=norm(label(b_ind)-H*x_closed,2)^2+mu*x_closed'*gamma*x_closed;
% % % % x_pred=sign(x_closed(1:1:n_sample));
% % % % err_count = sum(abs(sign(x_pred) - label))/2;
end

