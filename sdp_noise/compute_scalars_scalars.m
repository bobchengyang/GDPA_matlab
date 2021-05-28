function [ M_fv,...
    scaled_M,...
    scaled_factors] = ...
    compute_scalars_scalars( ...
    M,...
    M_fv,...
    dz_ind_plus,...
    dz_ind_minus,...
    n_sample)

M1=M([dz_ind_minus n_sample+1],[dz_ind_minus n_sample+1]);

[M_fv1,~] = ...
    lobpcg_fv(...
    M_fv([dz_ind_minus n_sample+1]),...
    M1,...
    1e-16,...
    1e3);

a1=M_fv1(:,1);

scaled_M1 = (1./a1) .* M1 .* a1';
scaled_factors1 = (1./a1) .* ones(length(M_fv1)) .* a1';

scaled_M_offdia1=scaled_M1;
dim1=size(scaled_M1,1);
scaled_M_offdia1(1:dim1+1:end)=0;
leftEnds1=diag(M1)-sum(abs(scaled_M_offdia1),2);
leftEnds_diff1=sum(abs(leftEnds1-mean(leftEnds1)));
disp(['v3 LP before LP LeftEnds group 1 mean: ' num2str(mean(leftEnds1)) ' | LeftEnds difference: ' num2str(leftEnds_diff1)]);

M2=M([dz_ind_plus n_sample+2],[dz_ind_plus n_sample+2]);

[M_fv2,~] = ...
    lobpcg_fv(...
    M_fv([dz_ind_plus n_sample+2]),...
    M2,...
    1e-16,...
    1e3);

a2=M_fv2(:,1);

scaled_M2 = (1./a2) .* M2 .* a2';
scaled_factors2 = (1./a2) .* ones(length(M_fv2)) .* a2';

scaled_M_offdia2=scaled_M2;
dim2=size(scaled_M2,1);
scaled_M_offdia2(1:dim2+1:end)=0;
leftEnds2=diag(M2)-sum(abs(scaled_M_offdia2),2);
leftEnds_diff2=sum(abs(leftEnds2-mean(leftEnds2)));
disp(['v3 LP before LP LeftEnds group 2 mean: ' num2str(mean(leftEnds2)) ' | LeftEnds difference: ' num2str(leftEnds_diff2)]);

all_idx=1:n_sample+2;
all_idx([dz_ind_plus dz_ind_minus n_sample+1 n_sample+2])=[];
M3=M(all_idx,all_idx);

[M_fv3,~] = ...
    lobpcg_fv(...
    M_fv(all_idx),...
    M3,...
    1e-16,...
    1e3);

a3=M_fv3(:,1);

scaled_M3 = (1./a3) .* M3 .* a3';
scaled_factors3 = (1./a3) .* ones(length(M_fv3)) .* a3';

scaled_M_offdia3=scaled_M3;
dim3=size(scaled_M3,1);
scaled_M_offdia3(1:dim3+1:end)=0;
leftEnds3=diag(M3)-sum(abs(scaled_M_offdia3),2);
leftEnds_diff3=sum(abs(leftEnds3-mean(leftEnds3)));
disp(['v3 LP before LP LeftEnds group 3 mean: ' num2str(mean(leftEnds3)) ' | LeftEnds difference: ' num2str(leftEnds_diff3)]);

scaled_M=zeros(n_sample+2);
scaled_M([dz_ind_minus n_sample+1],[dz_ind_minus n_sample+1])=scaled_M1;
scaled_M([dz_ind_plus n_sample+2],[dz_ind_plus n_sample+2])=scaled_M2;
scaled_M(all_idx,all_idx)=scaled_M3;



scaled_factors=zeros(n_sample+2);
scaled_factors([dz_ind_minus n_sample+1],[dz_ind_minus n_sample+1])=scaled_factors1;
scaled_factors([dz_ind_plus n_sample+2],[dz_ind_plus n_sample+2])=scaled_factors2;
scaled_factors(all_idx,all_idx)=scaled_factors3;

M_fv([dz_ind_minus n_sample+1])=M_fv1;
M_fv([dz_ind_plus n_sample+2])=M_fv2;
M_fv(all_idx)=M_fv3;
end

