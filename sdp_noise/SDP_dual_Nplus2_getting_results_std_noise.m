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
% cvx_solver SeDuMi % SDPT3 sometimes fails but always balanced; SeDuMi's may not be balanced!!!
cvx_solver Mosek
% cvx_solver SCS
cvx_begin quiet
cvx_precision low;

for dataset_i=1:1:17
    %%=======change the following things for experiments======
    [dataset_str,read_data] = get_data(dataset_i);
    % n_sample=min([size(read_data,1) 100]);% max total number of samples to be experimented
    rho=0; % PSD parameter
    experiment_save_str=['results_' dataset_str '_traintest_std_noise_default_82.mat'];
    %%========================================================
    
    num_run=100;
    results=zeros(num_run,62);
    
    label=read_data(:,end);
    if dataset_i~=17 && dataset_i~=5 && dataset_i~=7
    K=5; % 5-fold
    elseif dataset_i==17
    K=1; 
    elseif dataset_i==5
    K=4;
    elseif dataset_i==7
    K=2;
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
            b_ind = 1:1:round(0.8*n_sample); % every other sample / ~50% training sample
            [cL,feature,n_feature,label,label_noisy] = dataLoader_normalized_traintest_std_noise(read_data_i,n_sample,b_ind,rsrng);
            initial_label_index = logical(zeros(n_sample,1));
            initial_label_index(b_ind)=1;
            disp(['True labels: ' num2str(label(round((n_sample/2))+1:1:n_sample)')]);
            
            disp('KNN results====================================================================');
            [x_pred_knn,err_knn] = knn_classification(label,label_noisy,feature,n_feature,initial_label_index);
            disp(['KNN predicted labels: ' num2str(x_pred_knn')]);
            disp(['KNN error_count: ' num2str(err_knn)]);
            
            disp('Mahalanobis results====================================================================');
            [x_pred_maha,err_maha] = mahalanobis_classification(label,label_noisy,feature,n_feature,initial_label_index);
            disp(['Mahalanobis predicted labels: ' num2str(x_pred_maha')]);
            disp(['Mahalanobis error_count: ' num2str(err_maha)]);
            
            disp('SVM results====================================================================');
            [x_pred_svm,err_svm] = svm_classification(label,label_noisy,feature,initial_label_index);
            disp(['SVM predicted labels: ' num2str(x_pred_svm')]);
            disp(['SVM error_count: ' num2str(err_svm)]);
            
            disp('GLR quadratic results====================================================================');
            [x_pred_GLR_quad,error_count_GLR_quad,obj_GLR_quad,t_GLR_quad] = GLR_closed_form(...
                cL,...
                n_sample,...
                b_ind,...
                label,...
                label_noisy);
            disp(['GLR quadratic predicted labels: ' num2str(x_pred_GLR_quad')]);
            disp(['GLR quadratic error_count: ' num2str(error_count_GLR_quad)]);
            disp(['GLR quadratic obj: ' num2str(obj_GLR_quad)]);
            
            disp('closed form results====================================================================');
            gamma=trace(cL)*1e2;
            [y_new_eq_cf,current_obj_eq_cf,x_pred_eq_cf,err_count_eq_cf] = eq_closed_form_fast(gamma,label,label_noisy,n_sample,cL,b_ind);
            disp(['closed-form predicted labels: ' num2str(int8(x_pred_eq_cf'))]);
            disp(['closed-form error_count: ' num2str(err_count_eq_cf)]);
            disp(['closed-form obj: ' num2str(current_obj_eq_cf)]);
            
            %% Eq. max_{M} sum(vec(L.*M)) SDP standard form (interior-point)
            disp('original SDP PRIMAL results====================================================================');
            clear dB dA;
            dL = [-cL zeros(n_sample,1); zeros(1,n_sample) 0]; % eq. (6)
            ei_s = eye(n_sample);
            for i=1:length(b_ind)
                dB(:,:,i) = [zeros(n_sample,n_sample) ei_s(:,b_ind(i)); ei_s(b_ind(i),:) 0]; % eq. (7), where dB is a stacked matrix
            end
            [x_pred_eq8,error_count_eq8,obj_eq8,t_eq8] = eq8(dL,dB,n_sample,b_ind,label,label_noisy);
            
            disp(['original SDP PRIMAL predicted labels: ' num2str(int8(x_pred_eq8'))]);
            disp(['original SDP PRIMAL error_count: ' num2str(error_count_eq8)]);
            disp(['original SDP PRIMAL obj: ' num2str(obj_eq8)]);
            
            %% Eq. min_{y,z} sum(y)+sum(b.*z) SDP-dual (interior-point)
            disp('original SDP DUAL results====================================================================');
            ei_l = eye(n_sample+1);   %% two identity matrices
            for i=1:n_sample+1
                dA(:,:,i) = diag(ei_l(:,i)); % eq. (7), where dA is a stacked matrix where each of them has only 1 non-zero entry
            end
            [y_eq10,z_eq10,obj_eq10,db,x_pred_eq10,error_count_eq10,t_eq10,u,alpha] = eq10(label,label_noisy,b_ind,n_sample,dA,dB,dL);
            disp(['original SDP DUAL predicted labels: ' num2str(int8(x_pred_eq10'))]);
            disp(['original SDP DUAL error_count: ' num2str(error_count_eq10)]);
            disp(['original SDP DUAL y: ' num2str(vec(y_eq10)')]);
            disp(['original SDP DUAL z: ' num2str(vec(z_eq10)')]);
            disp(['original SDP DUAL obj: ' num2str(obj_eq10)]);
            
            disp('original SDP DUAL with cdcs results====================================================================');
            [y_eq10_cdcs,z_eq10_cdcs,obj_eq10_cdcs,db_cdcs,x_pred_eq10_cdcs,error_count_eq10_cdcs,...
                t_eq10_cdcs,u_cdcs,alpha_cdcs] = eq10_cdcs(label,label_noisy,b_ind,n_sample,dA,dB,dL);
            disp(['original SDP DUAL with cdcs predicted labels: ' num2str(int8(x_pred_eq10_cdcs'))]);
            disp(['original SDP DUAL with cdcs error_count: ' num2str(error_count_eq10_cdcs)]);
            disp(['original SDP DUAL with cdcs y: ' num2str(vec(y_eq10_cdcs)')]);
            disp(['original SDP DUAL with cdcs z: ' num2str(vec(z_eq10_cdcs)')]);
            disp(['original SDP DUAL with cdcs obj: ' num2str(obj_eq10_cdcs)]);
            
            disp('original SDP DUAL with cdcsd results====================================================================');
            [y_eq10_cdcsd,z_eq10_cdcsd,obj_eq10_cdcsd,db_cdcsd,x_pred_eq10_cdcsd,error_count_eq10_cdcsd,...
                t_eq10_cdcsd,u_cdcsd,alpha_cdcsd] = eq10_cdcs_d(label,label_noisy,b_ind,n_sample,dA,dB,dL);
            disp(['original SDP DUAL with cdcsd predicted labels: ' num2str(int8(x_pred_eq10_cdcsd'))]);
            disp(['original SDP DUAL with cdcsd error_count: ' num2str(error_count_eq10_cdcsd)]);
            disp(['original SDP DUAL with cdcsd y: ' num2str(vec(y_eq10_cdcsd)')]);
            disp(['original SDP DUAL with cdcsd z: ' num2str(vec(z_eq10_cdcsd)')]);
            disp(['original SDP DUAL with cdcsd obj: ' num2str(obj_eq10_cdcsd)]);
            
            disp('original SDP DUAL with sdcut results====================================================================');
            [y_eq10_sdcut,z_eq10_sdcut,obj_eq10_sdcut,db_sdcut,x_pred_eq10_sdcut,error_count_eq10_sdcut,...
                t_eq10_sdcut,u_sdcut,alpha_sdcut] = eq10_sdcut(label,label_noisy,b_ind,n_sample,dA,dB,dL);
            disp(['original SDP DUAL with sdcut predicted labels: ' num2str(int8(x_pred_eq10_sdcut'))]);
            disp(['original SDP DUAL with sdcut error_count: ' num2str(error_count_eq10_sdcut)]);
            disp(['original SDP DUAL with sdcut y: ' num2str(vec(y_eq10_sdcut)')]);
            disp(['original SDP DUAL with sdcut z: ' num2str(vec(z_eq10_sdcut)')]);
            disp(['original SDP DUAL with sdcut obj: ' num2str(obj_eq10_sdcut)]);
            
            %% Eq. SDP dual
            disp('v4 results====================================================================');
            [obj_direct,part_direct,x_pred_direct,err_count_direct,eigen_gap_direct,...
                dy_LP_test_init0,dz_LP_test_init0,new_H_LP_test_init0,...
                t_direct] = ...
                Nplus2_no_self_loop_c(label,label_noisy,b_ind,n_sample,dA,dB,dL);
            disp(['v4 predicted labels: ' num2str(int8(x_pred_direct'))]);
            disp(['v4 error_count: ' num2str(err_count_direct)]);
            disp(['v4 obj: ' num2str(obj_direct) ' | ' num2str(part_direct)]);
            disp(['v4 eigen gap: ' num2str(eigen_gap_direct)]);
            
            disp('v4 LP results==========================;==========================================');
            sw=0.5;
            u=0;
            alpha=0;
            [obj_direct_nic_LP,x_pred_direct_nic_LP,err_count_direct_nic_LP,u00_LP,alpha00_LP,eigen_gap_direct_nic_LP,...
                t_direct_nic_LP] = ...
                Nplus2_self_loop_c_GDPA(label,label_noisy,b_ind,n_sample,cL,u,alpha,sw,...
                rho);
            disp(['v4 LP predicted labels: ' num2str(int8(x_pred_direct_nic_LP'))]);
            disp(['v4 LP error_count: ' num2str(err_count_direct_nic_LP)]);
            disp(['v4 LP obj: ' num2str(obj_direct_nic_LP)]);
            disp(['v4 LP eigen gap: ' num2str(eigen_gap_direct_nic_LP)]);
            
            disp('v3 results====================================================================');
            alpha=1e15;
            sw=0.5;
            for i=1:1
                try
                    disp(['v3 iter '  num2str(i) '=======']);
                    [obj_direct_nii,x_pred_direct_nii,err_count_direct_nii,u_,alpha0,...
                        eigen_gap_direct_nii,t_direct_nii,...
                        dy_LP_test_init,dz_LP_test_init,new_H_LP_test_init] = ...
                        Nplus2_v3(label,label_noisy,b_ind,n_sample,dA,dB,dL,u,alpha,sw);
                    disp(['v3 predicted labels: ' num2str(int8(x_pred_direct_nii'))]);
                    disp(['v3 error_count: ' num2str(err_count_direct_nii)]);
                    disp(['v3 obj: ' num2str(obj_direct_nii)]);
                    disp(['v3 eigen gap: ' num2str(eigen_gap_direct_nii)]);
                catch
                    break
                end
            end
            
            disp('v3 cdcs results====================================================================');
            [obj_direct_nii_cdcs,x_pred_direct_nii_cdcs,err_count_direct_nii_cdcs,u_,alpha0,...
                eigen_gap_direct_nii_cdcs,t_direct_nii_cdcs,...
                dy_LP_test_init_cdcs,dz_LP_test_init_cdcs,new_H_LP_test_init_cdcs] = ...
                Nplus2_v3_cdcs(label,label_noisy,b_ind,n_sample,dA,dB,dL,u,alpha,sw);
            disp(['v3 cdcs predicted labels: ' num2str(int8(x_pred_direct_nii_cdcs'))]);
            disp(['v3 cdcs error_count: ' num2str(err_count_direct_nii_cdcs)]);
            disp(['v3 cdcs obj: ' num2str(obj_direct_nii_cdcs)]);
            disp(['v3 cdcs eigen gap: ' num2str(eigen_gap_direct_nii_cdcs)]);
            
            disp('v3 sdcut results====================================================================');
            [obj_direct_nii_sdcut,x_pred_direct_nii_sdcut,err_count_direct_nii_sdcut,u_,alpha0,...
                eigen_gap_direct_nii_sdcut,t_direct_nii_sdcut,...
                dy_LP_test_init_sdcut,dz_LP_test_init_sdcut,new_H_LP_test_init_sdcut] = ...
                Nplus2_v3_sdcut(label,label_noisy,b_ind,n_sample,dA,dB,dL,u,alpha,sw);
            disp(['v3 sdcut predicted labels: ' num2str(int8(x_pred_direct_nii_sdcut'))]);
            disp(['v3 sdcut error_count: ' num2str(err_count_direct_nii_sdcut)]);
            disp(['v3 sdcut obj: ' num2str(obj_direct_nii_sdcut)]);
            disp(['v3 sdcut eigen gap: ' num2str(eigen_gap_direct_nii_sdcut)]);
            
            disp('v3 LP results====================================================================');
            [obj_direct_nic_LP_v3,x_pred_direct_nic_LP_v3,err_count_direct_nic_LP_v3,u00_LP,alpha00_LP,eigen_gap_direct_nic_LP_v3,...
                t_direct_nic_LP_v3] = ...
                Nplus2_v3_LP(label,label_noisy,b_ind,n_sample,cL,u,alpha,sw,...
                rho);
            disp(['v3 LP predicted labels: ' num2str(int8(x_pred_direct_nic_LP_v3'))]);
            disp(['v3 LP error_count: ' num2str(err_count_direct_nic_LP_v3)]);
            disp(['v3 LP obj: ' num2str(obj_direct_nic_LP_v3)]);
            disp(['v3 LP eigen gap: ' num2str(eigen_gap_direct_nic_LP_v3)]);
            
            % disp('SDP_dual_balance ===================================');
            % N=n_sample;
            % M=length(b_ind);
            % SDP_dual_balance(N,M,label,cL,sw,alpha);
            
            % results=[obj_direct_nii t_direct_nii err_count_direct_nii err_count_direct_nii/length(find(~initial_label_index)) eigen_gap_direct_nii...
            %          obj_direct_nic_LP_v3 t_direct_nic_LP_v3 err_count_direct_nic_LP_v3 err_count_direct_nic_LP_v3/length(find(~initial_label_index)) eigen_gap_direct_nic_LP_v3...
            
            results(result_seq_i,:)=[obj_direct_nii t_direct_nii err_count_direct_nii err_count_direct_nii/length(find(~initial_label_index)) eigen_gap_direct_nii...
                obj_direct_nic_LP_v3 t_direct_nic_LP_v3 err_count_direct_nic_LP_v3 err_count_direct_nic_LP_v3/length(find(~initial_label_index)) eigen_gap_direct_nic_LP_v3...
                obj_direct_nii_cdcs t_direct_nii_cdcs err_count_direct_nii_cdcs err_count_direct_nii_cdcs/length(find(~initial_label_index)) eigen_gap_direct_nii_cdcs...
                obj_direct_nii_sdcut t_direct_nii_sdcut err_count_direct_nii_sdcut err_count_direct_nii_sdcut/length(find(~initial_label_index)) eigen_gap_direct_nii_sdcut...
                obj_direct     t_direct err_count_direct     err_count_direct/length(find(~initial_label_index))  eigen_gap_direct...
                obj_direct_nic_LP t_direct_nic_LP err_count_direct_nic_LP err_count_direct_nic_LP/length(find(~initial_label_index)) eigen_gap_direct_nic_LP...
                obj_eq10       t_eq10 error_count_eq10 error_count_eq10/length(find(~initial_label_index))...
                obj_eq10_cdcs  t_eq10_cdcs error_count_eq10_cdcs error_count_eq10_cdcs/length(find(~initial_label_index))...
                obj_eq10_cdcsd  t_eq10_cdcsd error_count_eq10_cdcsd error_count_eq10_cdcsd/length(find(~initial_label_index))...
                obj_eq10_sdcut t_eq10_sdcut error_count_eq10_sdcut error_count_eq10_sdcut/length(find(~initial_label_index))...
                obj_eq8        t_eq8 error_count_eq8 error_count_eq8/length(find(~initial_label_index))...
                err_count_eq_cf err_count_eq_cf/length(find(~initial_label_index))...
                obj_GLR_quad t_GLR_quad error_count_GLR_quad error_count_GLR_quad/length(find(~initial_label_index))...
                err_knn err_knn/length(find(~initial_label_index))...
                err_maha err_maha/length(find(~initial_label_index))...
                err_svm err_svm/length(find(~initial_label_index))];
        end
    end
    
    clearvars -except dataset_i experiment_save_str results
    cvx_precision low;
    save(experiment_save_str);
end