clear;clc;close all;

[A,label] = dataLoader('monk1.csv');

data = A; % adjcency matrix
n_sample = 10; % number of total sample (this is a subset of the original dataset, just for experimentation purpose)
data = data(1:n_sample,1:n_sample); % total data
label = label(1:n_sample); % total label
D = diag(sum(data,2)); % degree matrix
cL = D - data; % combinatorial graph Laplacian

b_ind = 1:2:n_sample; % every other sample / ~50% training sample

% %% Dual formulations ****************************************************
db = 2*label(b_ind); % training labels x 2
dL = [-cL zeros(n_sample,1); zeros(1,n_sample) 0]; % eq. (7)
ei_l = eye(n_sample+1);   %% two identity matrices
ei_s = eye(n_sample);
for i=1:n_sample+1
    dA(:,:,i) = diag(ei_l(:,i)); % eq. (8), where dA is a stacked matrix where each of them has only 1 non-zero entry
end
for i=1:length(b_ind)
    dB(:,:,i) = [zeros(n_sample,n_sample) ei_s(:,b_ind(i)); ei_s(b_ind(i),:) 0]; % eq. (8), where dB is a stacked matrix
end


%% GDA version

tol=Inf;
obj_tol=1e-5;
main_iter=0;
while tol>obj_tol
    
    if tol==Inf
        % step 1: Initialize a version of the PSD matrix
        b_ind_logical=zeros(n_sample,1);
        b_ind_logical(b_ind)=1;
        
        %======INITIALIZE A PSD MATRIX ABL======
        y_initial_factor=.1;
        z_initial_factor=.1/n_sample;
        ABL=eye(n_sample+1)*y_initial_factor...
            +[zeros(n_sample,n_sample) -ones(n_sample,1)*z_initial_factor.*b_ind_logical; ...
            -ones(1,n_sample)*z_initial_factor.*b_ind_logical' 0]...
            -dL; % A+B-L
        %======INITIALIZE A PSD MATRIX ABL======
        
        M_current_eigenvector0=randn(size(ABL,1),1);
        y=ones(n_sample+1,1)*y_initial_factor;
        z=-ones(length(b_ind),1)*z_initial_factor;
    end
    
    obj=ones(1,n_sample+1)*y + db'*z;
    main_iter=main_iter+1;
    disp(['main iteration ' num2str(main_iter) ' | current obj: ' num2str(obj) ' | mineig: ' num2str(min(eig(ABL)))]);
    
    % step 2: compute the scalars of ABL
    [M_current_eigenvector0,...
        scaled_M,...
        scaled_factors] = ...
        compute_scalars(...
        ABL,...
        M_current_eigenvector0);
    
    % step 3: find optimal y and z via LP (used linprog)
    options = optimoptions('linprog','Display','none','Algorithm','interior-point'); % linear program (LP) setting for Frank-Wolfe algorithm
    % options=optimoptions('linprog','Algorithm','interior-point'); % linear program (LP) setting for Frank-Wolfe algorithm
    options.OptimalityTolerance=1e-5; % LP optimality tolerance
    options.ConstraintTolerance=1e-5; % LP interior-point constraint tolerance
    rho=1e-5; % PSD tol
    
    %% DIAGONAL
    for i=1:n_sample+1
        [y0]=LP_solve_dia( ...
            n_sample,...
            scaled_M,...
            rho,...
            dL,...
            i,...
            options);  
          
        %====== update ABL ======
        y(i)=y0; % update y
        dM=zeros(n_sample+1);
        for j=1:n_sample+1
            dM = dM + y(j)*dA(:,:,j);
        end
        for j=1:length(b_ind)
            dM = dM + z(j)*dB(:,:,j);
        end
        ABL=dM-dL;
        %====== update ABL ======  
        
        obj0=ones(1,n_sample+1)*y + db'*z;
        
        %====== update scalars ======
        [M_current_eigenvector0,...
            scaled_M,...
            scaled_factors] = ...
            compute_scalars(...
            ABL,...
            M_current_eigenvector0);
        %====== update scalars ======
    end
    
    %% OFF-DIAGONAL
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
        
        %====== update ABL ======
        z(counter)=z0; % update z
        dM=zeros(n_sample+1);
        for j=1:n_sample+1
            dM = dM + y(j)*dA(:,:,j);
        end
        for j=1:length(b_ind)
            dM = dM + z(j)*dB(:,:,j);
        end
        ABL=dM-dL;
        %====== update ABL ======  
        
        obj0=ones(1,n_sample+1)*y + db'*z;
   
        % check balance of the LP solution (Dinesh kindly sent the following BFS_Balanced code to me)
        ABL=-BFS_Balanced(-ABL);        

        %====== update scalars ======
        [M_current_eigenvector0,...
            scaled_M,...
            scaled_factors] = ...
            compute_scalars(...
            ABL,...
            M_current_eigenvector0);
        %====== update scalars ======
        
        end
    end
    
    tol=norm(obj-(ones(1,n_sample+1)*y + db'*z));
    
end

y
z