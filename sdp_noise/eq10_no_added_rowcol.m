% function [dy,dz,cvx_optval,db,x_pred,err_count,LA_obj] = eq10(label,b_ind,n_sample,dA,dB,dL,mu)
function [dy,dz,cvx_optval,db,x_pred,err_count] = eq10_no_added_rowcol(label,b_ind,n_sample,dA,dB,dL)
db = 2*label(b_ind)*label(b_ind(1)); % training labels x 2
dM = zeros(n_sample,n_sample); % number of total samples + 1
cvx_begin sdp
variables dy(n_sample,1) dz(length(b_ind)-1,1);
minimize(sum(dy) + db(2:end)'*dz)
subject to
for i=1:n_sample
    dM = dM + dy(i)*dA(:,:,i);
end
for i=2:length(b_ind)
    dM = dM + dz(i-1)*dB(:,:,i);
end
dM-dL>=0
% dM>=0
cvx_end

%% check graph balanceness of H_tilde

H=dM-dL;
H=(H+H')/2;
H_0=BFS_Balanced(-H);
if isequal(H_0,-H)
    disp('H balanced.');
else
    disp('H NOT balanced.');
end

full_dM=full(dM);
full_dM = (full_dM+full_dM')/2;

fv_H_0=randn(n_sample,1);

[fv_H,lambda] = ...
    lobpcg_fv(...
    fv_H_0,...
    H,...
    1e-4,...
    200);

%x_val = [sign(label(b_ind(1)));sign(-dz*label(b_ind(1)))];
x_val = sign(label(b_ind(1))*sign(fv_H(1)))*sign(fv_H(1:n_sample));
% x_val = sign(fv_H(1:n_sample));
x_pred=sign(x_val(round(n_sample/2)+1:1:n_sample));
err_count = sum(abs(sign(x_val) - label))/2;

H_tilde=H(2:end,2:end);
gamma=H_tilde-(1/H(1,1))*H(2:end,1)*H(2:end,1)';
gamma=(gamma+gamma')/2;
gamma_0=BFS_Balanced(-gamma);
if isequal(gamma_0,-gamma)
    disp('gamma balanced.');
else
    disp('gamma NOT balanced.');
end

% % % % ly=-dL(2:end,2:end);
% % % % G=zeros(n_sample-1)-2*1e-2;
% % % % G(1:n_sample-1+1:end)=diag(ly)*-dL(1,1)*0.5;
% % % % 
% % % % ma=[ly-(-dL(1,1))*G zeros(n_sample-1); zeros(n_sample-1) G];
% % % % 
% % % % ul_ma=ma(1:n_sample-1,1:n_sample-1);
% % % % br_ma=ma(n_sample-1+1:end,n_sample-1+1:end);
% % % % 
% % % % fv_ulma_0=randn(n_sample-1,1);
% % % % 
% % % % [fv_ulma,...
% % % %     scaled_ulma,...
% % % %     scaled_factors_ulma] = ...
% % % %     compute_scalars(...
% % % %     ul_ma,...
% % % %     fv_ulma_0); % compute scalars
% % % % 
% % % % scaled_ulma_offdia=scaled_ulma;
% % % % scaled_ulma_offdia(1:n_sample-1+1:end)=0;
% % % % leftEnds_ulma=diag(ul_ma)-sum(abs(scaled_ulma_offdia),2);
% % % % leftEnds_diff_ulma=sum(abs(leftEnds_ulma-mean(leftEnds_ulma)));
% % % % disp(['before LP LeftEnds_ulma mean: ' num2str(mean(leftEnds_ulma)) ' | LeftEnds difference: ' num2str(leftEnds_diff_ulma)]);
% % % % 
% % % % fv_brma_0=randn(n_sample-1,1);
% % % % 
% % % % [fv_brma,...
% % % %     scaled_brma,...
% % % %     scaled_factors_brma] = ...
% % % %     compute_scalars(...
% % % %     br_ma,...
% % % %     fv_brma_0); % compute scalars
% % % % 
% % % % scaled_brma_offdia=scaled_brma;
% % % % scaled_brma_offdia(1:n_sample-1+1:end)=0;
% % % % leftEnds_brma=diag(br_ma)-sum(abs(scaled_brma_offdia),2);
% % % % leftEnds_diff_brma=sum(abs(leftEnds_brma-mean(leftEnds_brma)));
% % % % disp(['before LP LeftEnds_brma mean: ' num2str(mean(leftEnds_brma)) ' | LeftEnds difference: ' num2str(leftEnds_diff_brma)]);


% sign(dz*dz')+sign(gamma)
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

