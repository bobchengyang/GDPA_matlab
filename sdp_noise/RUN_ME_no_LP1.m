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


%% GDA no LP
disp('==============');
disp('GDA no LP started.');
tol=Inf;
obj_tol=1e-5;
main_iter=0;
while tol>obj_tol
    
    if tol==Inf
        % step 1: Initialize a version of the PSD matrix
        b_ind_logical=zeros(n_sample,1);
        b_ind_logical(b_ind)=1;
        
        %======INITIALIZE A PSD MATRIX ABL======
        y_initial_0=zeros(n_sample,1);
        y_initial_0(b_ind)=1;
        y_initial=[y_initial_0;sum(y_initial_0)];
        z_initial_factor=.1;
        ABL=diag(y_initial)...
            +[zeros(n_sample,n_sample) -ones(n_sample,1)*z_initial_factor.*b_ind_logical; ...
            -ones(1,n_sample)*z_initial_factor.*b_ind_logical' 0]...
            -dL; % A+B-L, the matrix in the constriant of eq. (11)
        %======INITIALIZE A PSD MATRIX ABL======
        rng(0);
        M_current_eigenvector0=randn(size(ABL,1),1);
        y=y_initial;
        z=-ones(length(b_ind),1)*z_initial_factor;
    end
    
    obj=ones(1,n_sample+1)*y + db'*z;
    main_iter=main_iter+1;
    disp(['GDA no LP main iteration ' num2str(main_iter) ' | current obj: ' num2str(obj) ' | mineig: ' num2str(min(eig(ABL)))]);
    
    % step 2: compute the scalars of ABL
    [M_current_eigenvector0,...
        scaled_M,...
        scaled_factors] = ...
        compute_scalars(...
        ABL,...
        M_current_eigenvector0);
    
    % step 3: find optimal y and z
    rho=1e-5; % PSD tol
    
    %% DIAGONAL one by one
    scaled_M_offdia=scaled_M;
    scaled_M_offdia(1:n_sample+1+1:end)=0;
    radii=sum(abs(scaled_M_offdia),2)+rho;
    y=radii-[diag(cL);0];
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
    %====== update scalars ======
    [M_current_eigenvector0,...
        scaled_M,...
        scaled_factors] = ...
        compute_scalars(...
        ABL,...
        M_current_eigenvector0);
    %====== update scalars ======
    
    %% OFF-DIAGONAL / DIAGONAL one by one
    counter=0;
    for i=1:n_sample+1
        if ismember(i,b_ind)
            counter=counter+1;
            remaning_idx=1:n_sample+1;
            remaning_idx([i n_sample+1])=[];
            nn_1=scaled_M(i,i)-sum(abs(scaled_M(i,remaning_idx)))-rho;
            remaning_idx=1:n_sample;
            remaning_idx(i)=[];
            dn_1=scaled_M(n_sample+1,n_sample+1)-sum(abs(scaled_M(n_sample+1,remaning_idx)))-rho;
            nn_end=abs(scaled_factors(i,n_sample+1));
            dn_end=abs(scaled_factors(n_sample+1,i));
            abs_z0_1=nn_1/dn_1;
            abs_z0_end=nn_end/dn_end;
            if abs_z0_1<=0 || abs_z0_end<=0
                asdf=1;
            else
                z0=-min([abs_z0_1 abs_z0_end]);
            end
            %====== update ABL ======
            if ~isequal(z0,Inf)
                z(counter)=z0; % update z
            end
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
                ~] = ...
                compute_scalars(...
                ABL,...
                M_current_eigenvector0);
            %====== update scalars ======
            
            scaled_M_offdia=scaled_M;
            scaled_M_offdia(1:n_sample+1+1:end)=0;
            radii=sum(abs(scaled_M_offdia),2)+rho;
            y=radii-[diag(cL);0];
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
disp('=================');
disp('GDA no LP converged.');
disp('GDA no LP solutions:');
y
z