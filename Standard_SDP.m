clear;clc;close all;

addpath('e:\08-AUG-2020 sdp\icassp sdp\YALMIP-master\');
addpath('e:\08-AUG-2020 sdp\icassp sdp\YALMIP-master\extras\');
addpath('e:\08-AUG-2020 sdp\icassp sdp\YALMIP-master\solvers\');
addpath('e:\08-AUG-2020 sdp\icassp sdp\YALMIP-master\modules\');
addpath('e:\08-AUG-2020 sdp\icassp sdp\YALMIP-master\modules\parametric\');
addpath('e:\08-AUG-2020 sdp\icassp sdp\YALMIP-master\modules\moment\');
addpath('e:\08-AUG-2020 sdp\icassp sdp\YALMIP-master\modules\global\');
addpath('e:\08-AUG-2020 sdp\icassp sdp\YALMIP-master\modules\robust\');
addpath('e:\08-AUG-2020 sdp\icassp sdp\YALMIP-master\modules\sos\');
addpath('e:\08-AUG-2020 sdp\icassp sdp\YALMIP-master\operators\');
% cvx_begin quiet;
%addpath(genpath(yalmiprootdirectory))

n_sample = 20; % number of total sample (this is a subset of the original dataset, just for experimentation purpose)

% % % % [A,label] = dataLoader('monk1.csv',n_sample);
% % % % %data = [0 1 10;1 0 3;10 3 0];
% % % % %data = [0 1 2; 1 0 3;2 3 0];
% % % % data = A; % adjcency matrix
% % % % data = data(1:n_sample,1:n_sample); % total data
% % % % label = label(1:n_sample); % total label
% % % % D = diag(sum(data,2)); % degree matrix
% % % % cL = D - data; % combinatorial graph Laplacian
% % % % [f,p] = eig(cL); % eigen-decomposition of cL

[cL,label] = dataLoader_normalized('monk1.csv',n_sample);

dia_idx=(1:n_sample+1:n_sample^2)'; % get the indices of the diagonal entries
b_ind = 1:2:n_sample; % every other sample / ~50% training sample
m_ind = 1:n_sample^2; %

%% Primal formulations *************************************************
% X = sdpvar(n_sample,n_sample);
% x = sdpvar(n_sample,1);
% t = trace(cL*X);

%% old SDP relaxation in Cheng's APSIPA'18 ***************
% M = [X x; x' 1];

%% new SDP relaxation in our current writeup *************
% M = [-X+2*n_sample*eye(n_sample) x; x' 1];

%% call SDP optimization %%%
cvx_begin sdp
variables X(n_sample,n_sample) x(n_sample,1);
minimize(trace(cL*X))
subject to
[X x; x' 1]>=0; 
X(dia_idx) == 1; 
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
err_count = sum(abs(sign(x_val) - label))/2


% %% Dual formulations ****************************************************
% dy = sdpvar(n_sample+1,1); % its length is equal to the number of total samples + 1
% dz = sdpvar(length(b_ind),1); % its length is equal to the number of training samples
db = 2*label(b_ind); % training labels x 2
% t = ones(1,n_sample+1)*dy + db'*dz; % dual objective
%
dL = [-cL zeros(n_sample,1); zeros(1,n_sample) 0]; % eq. (7)
ei_l = eye(n_sample+1);   %% two identity matrices
ei_s = eye(n_sample);
for i=1:n_sample+1
    dA(:,:,i) = diag(ei_l(:,i)); % eq. (8), where dA is a stacked matrix where each of them has only 1 non-zero entry
end
for i=1:length(b_ind)
    dB(:,:,i) = [zeros(n_sample,n_sample) ei_s(:,b_ind(i)); ei_s(b_ind(i),:) 0]; % eq. (8), where dB is a stacked matrix
end
dM = zeros(n_sample+1,n_sample+1); % number of total samples + 1
% for i=1:n_sample+1
%     dM = dM + dy(i)*dA(:,:,i);
% end
% for i=1:length(b_ind)
%     dM = dM + dz(i)*dB(:,:,i);
% end
% F = [dM>=0];
% optimize(F,t,sdpsettings('solver','sedumi'));
label_b_ind=label(b_ind);
cvx_begin sdp
variables M(n_sample+1,n_sample+1);
minimize(-sum(sum(dL.*M)))
subject to
M(1:n_sample+1+1:end)==1;
for i=1:length(b_ind)
    sum(sum((dB(:,:,i).*M)))==2*label_b_ind(i);
end
M>=0
cvx_end

x_val=M(1:end-1,end);
err_count = sum(abs(sign(x_val) - label))/2

cvx_begin sdp
variables dy(n_sample+1,1) dz(length(b_ind),1);
minimize(ones(1,n_sample+1)*dy + db'*dz)
subject to
for i=1:n_sample+1
    dM = dM + dy(i)*dA(:,:,i);
end
for i=1:length(b_ind)
    dM = dM + dz(i)*dB(:,:,i);
end
dM-dL>=0
cvx_end

dy
dz

% t_val = value(t);
% y_val = value(dy);
% z_val = value(dz);
% %err_count = sum(abs(sign(x_val) - label))/2

%% GDA version

tol=Inf;
obj_tol=1e-5;
main_iter=0;
while tol>obj_tol
    
    if tol==Inf
        % step 1: Initialize a version of the PSD matrix
        b_ind_logical=zeros(n_sample,1);
        b_ind_logical(b_ind)=1;
        y_initial_factor=20;
        z_initial_factor=20;
        initial_dM=eye(n_sample+1)*y_initial_factor...
            +[zeros(n_sample,n_sample) -ones(n_sample,1)*z_initial_factor.*b_ind_logical; ...
            -ones(1,n_sample)*z_initial_factor.*b_ind_logical' 0]...
            -dL; % A+B-L
        M_current_eigenvector0=randn(size(initial_dM,1),1);
        y=ones(n_sample+1,1)*y_initial_factor;
        z=-ones(length(b_ind),1)*z_initial_factor;
%         initial_dM=dM-dL;
%         y=dy;
%         z=dz;
    end
    
    % check min eigenvalue of initial_dM
    obj=ones(1,n_sample+1)*y + db'*z;
    main_iter=main_iter+1;
    disp(['main iteration ' num2str(main_iter) ' | current obj: ' num2str(obj) ' | mineig: ' num2str(min(eig(initial_dM)))]);
    
    % step 2: compute the scalars of initial_dM
    [M_current_eigenvector0,...
        scaled_M,...
        scaled_factors] = ...
        compute_scalars(...
        initial_dM,...
        M_current_eigenvector0);
    
    % step 3: find optimal y and z via LP (used linprog)
    options = optimoptions('linprog','Display','none','Algorithm','interior-point'); % linear program (LP) setting for Frank-Wolfe algorithm
    % options=optimoptions('linprog','Algorithm','interior-point'); % linear program (LP) setting for Frank-Wolfe algorithm
    options.OptimalityTolerance=1e-5; % LP optimality tolerance
    options.ConstraintTolerance=1e-5; % LP interior-point constraint tolerance
    rho=1e-5; % PSD tol
    
    %% update the diagonal entries one by one
    for i=1:n_sample+1
        [y0]=LP_solve_dia( ...
            n_sample,...
            scaled_M,...
            rho,...
            dL,...
            i,...
            options);
        y(i)=y0; % update y
        dM=zeros(n_sample+1);
        for j=1:n_sample+1
            dM = dM + y(j)*dA(:,:,j);
        end
        for j=1:length(b_ind)
            dM = dM + z(j)*dB(:,:,j);
        end
        initial_dM=dM-dL;
        obj0=ones(1,n_sample+1)*y + db'*z;
%         disp(['current obj: ' num2str(obj0) ' | mineig: ' num2str(min(eig(initial_dM))) ' | dia ' num2str(i)]);   
        % update scalars
        [M_current_eigenvector0,...
            scaled_M,...
            scaled_factors] = ...
            compute_scalars(...
            initial_dM,...
            M_current_eigenvector0);
    end
    
    %% update the off-diagonals one by one
    counter=0;
    for i=1:n_sample+1
        if ismember(i,b_ind)
            counter=counter+1;
        [z0]=LP_solve_offdia( ...
            n_sample,...
            scaled_M,...
            scaled_factors,...
            rho,...
            i,...
            db(counter),...
            options);
        z(counter)=z0; % update z
        dM=zeros(n_sample+1);
        for j=1:n_sample+1
            dM = dM + y(j)*dA(:,:,j);
        end
        for j=1:length(b_ind)
            dM = dM + z(j)*dB(:,:,j);
        end
        initial_dM=dM-dL;
        obj0=ones(1,n_sample+1)*y + db'*z;
%         disp(['current obj: ' num2str(obj0) ' | mineig: ' num2str(min(eig(initial_dM))) ' | offdia ' num2str(counter) ' out of ' num2str(length(b_ind))]);   
        % update scalars
        [M_current_eigenvector0,...
            scaled_M,...
            scaled_factors] = ...
            compute_scalars(...
            initial_dM,...
            M_current_eigenvector0);
        end
    end
  
    % step 4: check balance of the LP solution
    
    tol=norm(obj-(ones(1,n_sample+1)*y + db'*z));

end

y
z














