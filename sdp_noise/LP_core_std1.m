clear;clc;close all;

[A,label] = dataLoader('monk1.csv');

%% ===== set initial y's z's to compute the scalars ===
y_set=1;
z_set=1;
%% ====================================================

data = A; % adjcency matrix
n_sample = 4; % number of total sample (this is a subset of the original dataset, just for experimentation purpose)
data = data(1:n_sample,1:n_sample); % total data
label = label(1:n_sample); % total label
D = diag(sum(data,2)); % degree matrix
cL = D - data; % combinatorial graph Laplacian

b_ind = 1:2:n_sample; % every other sample / ~50% training sample

% %% Dual formulations ****************************************************
db = 2*label(b_ind); % training labels x 2
dL = [-cL zeros(n_sample,1); zeros(1,n_sample) 0]; % eq. (6)
ei_l = eye(n_sample+1);   %% two identity matrices
ei_s = eye(n_sample);
for i=1:n_sample+1
    dA(:,:,i) = diag(ei_l(:,i)); % eq. (7), where dA is a stacked matrix where each of them has only 1 non-zero entry
end
for i=1:length(b_ind)
    dB(:,:,i) = [zeros(n_sample,n_sample) ei_s(:,b_ind(i)); ei_s(b_ind(i),:) 0]; % eq. (7), where dB is a stacked matrix
end

%% GDA LP
disp('==============');
disp('GDA LP started.');
tol=Inf;
obj_tol=1e-10;
main_iter=0;

while tol>obj_tol
    
    if tol==Inf
        % step 1: Initialize a version of the PSD matrix
        b_ind_logical=zeros(n_sample,1);
        b_ind_logical(b_ind)=1;
        
        %======INITIALIZE A PSD MATRIX ABL======
%         y_initial_0=zeros(n_sample,1);
%         y_initial_0(b_ind)=y_set;
%         y_initial=[y_initial_0;sum(y_initial_0)];
%         z_initial_factor=z_set;
%         ABL=diag(y_initial)...
%             +[zeros(n_sample,n_sample) -ones(n_sample,1)*z_initial_factor.*b_ind_logical; ...
%             -ones(1,n_sample)*z_initial_factor.*b_ind_logical' 0]...
%             -dL; % A+B-L, the matrix in the constriant of eq. (11)
        y=[ones(n_sample,1)*y_set;sum(ones(n_sample,1)*y_set)];
        z=ones(length(b_ind),1)*z_set;
        ABL=diag(y)+[zeros(n_sample,n_sample) ones(n_sample,1)*z_set.*b_ind_logical; ...
            ones(1,n_sample)*z_set.*b_ind_logical' 0]-dL;
        %======INITIALIZE A PSD MATRIX ABL======
        rng(0);
        M_current_eigenvector0=randn(size(ABL,1),1);
%         y=y_initial;
%         z=-ones(length(b_ind),1)*z_initial_factor;
    end    
    obj=ones(1,n_sample+1)*y + db'*z;
    main_iter=main_iter+1;
    disp(['GDA LP main iteration ' num2str(main_iter) ' | current obj: ' num2str(obj) ' | mineig: ' num2str(min(eig(ABL)))]);
    
    % step 2: compute the scalars of ABL
    [M_current_eigenvector0,...
        scaled_M,...
        scaled_factors] = ...
        compute_scalars(...
        ABL,...
        M_current_eigenvector0);
    
    % step 3: find optimal y and z via LP (used linprog)
%     options = optimoptions('linprog','Display','none','Algorithm','interior-point'); % linear program (LP) setting for Frank-Wolfe algorithm
    options=optimoptions('linprog','Algorithm','interior-point'); % linear program (LP) setting for Frank-Wolfe algorithm
    options.OptimalityTolerance=1e-5; % LP optimality tolerance
    options.ConstraintTolerance=1e-5; % LP interior-point constraint tolerance
    rho=1e-5; % PSD tol
        
    %% optimize y and z
    [y,z,obj_LP]=LP_solve_eq_LP_std1( ...
        n_sample,...
        scaled_M,...
        scaled_factors,...
        rho,...
        db,...
        options,...
        -dL,...
        b_ind);
        %====== update ABL ======
        dM=zeros(n_sample+1);
        for j=1:n_sample+1
            dM = dM + y(j)*dA(:,:,j);
        end
        for j=1:length(b_ind)
            dM = dM + z(j)*dB(:,:,j);
        end
        ABL=dM-dL;
        %====== update ABL ======  
        disp(['GDA LP main iteration ' num2str(main_iter) ' | current obj: ' num2str(obj) ' | pre-balancing mineig: ' num2str(min(eig(ABL)))]);
        
        %====== check balance ======
%         ABL=-BFS_Balanced(-ABL);    
        %====== check balance ======
        
        %====== update scalars ======
        [M_current_eigenvector0,...
            scaled_M,...
            scaled_factors] = ...
            compute_scalars(...
            ABL,...
            M_current_eigenvector0);
        %====== update scalars ======    

    tol=norm(obj-(ones(1,n_sample+1)*y + db'*z));  
end
disp('=================');
disp('GDA LP converged.');
disp('GDA LP solutions:');
y
z