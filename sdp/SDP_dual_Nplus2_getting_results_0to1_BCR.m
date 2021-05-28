clear all;
clc;
close all;
rng('default');

% cvx_precision low: [ϵ3/8,ϵ1/4,ϵ1/4]
% cvx_precision medium: [ϵ1/2,ϵ3/8,ϵ1/4]
% cvx_precision default: [ϵ1/2,ϵ1/2,ϵ1/4]
% cvx_precision high: [ϵ3/4,ϵ3/4,ϵ3/8]
% cvx_precision best: [0,ϵ1/2,ϵ1/4]

addpath('D:\Program Files\cvx-w64\cvx');
addpath('E:\08-AUG-2020 sdp\icassp sdp\CDCS-master\CDCS-master');
addpath('E:\08-AUG-2020 sdp\icassp sdp\L-BFGS-B-C-master\L-BFGS-B-C-master\Matlab');
% cvx_solver SDPT3 % SDPT3 sometimes fails but always balanced; SeDuMi's may not be balanced!!!
cvx_solver SeDuMi % SDPT3 sometimes fails but always balanced; SeDuMi's may not be balanced!!!
% cvx_solver Mosek
% cvx_solver SCS
cvx_begin quiet
cvx_precision low;

for dataset_i=1:17
    %%=======change the following things for experiments======
    [dataset_str,read_data] = get_data(dataset_i);
    % n_sample=min([size(read_data,1) 100]);% max total number of samples to be experimented
    rho=0; % PSD parameter
    experiment_save_str=['results_' dataset_str '_0to1_latest_settings_default_add2.mat'];
    %%========================================================
    
    num_run=100;
    results=zeros(num_run,4);
    
    label=read_data(:,end);
    if dataset_i~=17
    K=5; % 5-fold
    else
    K=1;   
    end
    rng(0);
    indices = crossvalind('Kfold',label,K); % K-fold cross-validation
    result_seq_i=0;
    for fold_i=1:K
        read_data_i=read_data(indices==fold_i,:);
        n_sample=size(read_data_i,1);
        for rsrng=1:10
            result_seq_i=result_seq_i+1;
            disp('==============================================================');
            disp(['======================= dataset ' num2str(dataset_i) ' fold ' num2str(fold_i) ' run number: ' num2str(rsrng) ' =======================']);
            disp('==============================================================');
            disp('==============================================================');
            b_ind = 1:1:round(0.5*n_sample); % every other sample / ~50% training sample
            [cL,feature,n_feature,label] = dataLoader_normalized_0to1(read_data_i,n_sample,b_ind,rsrng);
            initial_label_index = logical(zeros(n_sample,1));
            initial_label_index(b_ind)=1;
            disp(['True labels: ' num2str(label(round((n_sample/2))+1:1:n_sample)')]);

            %% Eq. max_{M} sum(vec(L.*M)) SDP standard form (interior-point)
            disp('original SDP PRIMAL results====================================================================');
            clear dB dA;
            dL = [-cL zeros(n_sample,1); zeros(1,n_sample) 0]; % eq. (6)
            ei_s = eye(n_sample);
            for i=1:length(b_ind)
                dB(:,:,i) = [zeros(n_sample,n_sample) ei_s(:,b_ind(i)); ei_s(b_ind(i),:) 0]; % eq. (7), where dB is a stacked matrix
            end
            [x_pred_eq8,error_count_eq8,obj_eq8,t_eq8] = eq8(dL,dB,n_sample,b_ind,label);
            
            disp(['original SDP PRIMAL predicted labels: ' num2str(int8(x_pred_eq8'))]);
            disp(['original SDP PRIMAL error_count: ' num2str(error_count_eq8)]);
            disp(['original SDP PRIMAL obj: ' num2str(obj_eq8)]);            

            disp('original SDP PRIMAL with BCR results====================================================================');
            ei_l = eye(n_sample+1);   %% two identity matrices
            for i=1:n_sample+1
                dA(:,:,i) = diag(ei_l(:,i)); % eq. (7), where dA is a stacked matrix where each of them has only 1 non-zero entry
            end            
            [error_count_eq10_bcr,...
                t_eq10_bcr] = eq10_BCR(label,b_ind,n_sample,dA,dB,dL);
            disp(['original SDP PRIMAL with BCR error_count: ' num2str(error_count_eq10_bcr)]);
            
            results(result_seq_i,:)=[t_eq8 error_count_eq8/length(find(~initial_label_index))...
                                     t_eq10_bcr error_count_eq10_bcr/length(find(~initial_label_index))];
        end
    end
    
    clearvars -except dataset_i experiment_save_str results
    cvx_precision low;
    save(experiment_save_str);
end