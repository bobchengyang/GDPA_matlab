function [y_new,current_obj,x_pred,err_count] = eq_same_label(label,n_sample,cL)
%eq_no_label Summary of this function goes here
%   Detailed explanation goes here

%% initialize a psd matrix ABL
rng(0);
y_0=zeros(n_sample,1);

ABL=diag(y_0)+cL; % initialize ABL

rng(0);
fv_ABL=randn(n_sample,1);

initial_obj=sum(y_0);
disp(['Eq. no label GDA LP main iteration ' num2str(0) ' | current obj: ' num2str(initial_obj) ' | mineig: ' num2str(min(eig(ABL)))]);

tol_set=1e-5;
tol=Inf;
loop_i=0;
while tol>tol_set
    if loop_i==0
    [fv_ABL,...
        scaled_M,...
        scaled_factors] = ...
        compute_scalars(...
        ABL,...
        fv_ABL); % compute scalars
    
    scaled_M_offdia=scaled_M;
    scaled_M_offdia(1:n_sample+1:end)=0;
    leftEnds=diag(ABL)-sum(abs(scaled_M_offdia),2);
    leftEnds_diff=sum(abs(leftEnds-mean(leftEnds)));
    disp(['Eq. no label before LP LeftEnds mean: ' num2str(mean(leftEnds)) ' | LeftEnds difference: ' num2str(leftEnds_diff)]);
    end
    
    y_new=-leftEnds;
    
    ABL=diag(y_new)+cL; % initialize ABL
    
    [fv_ABL,...
        scaled_M,...
        scaled_factors] = ...
        compute_scalars(...
        ABL,...
        fv_ABL); % compute scalars
    scaled_M_offdia=scaled_M;
    scaled_M_offdia(1:n_sample+1:end)=0;
    leftEnds=diag(ABL)-sum(abs(scaled_M_offdia),2);
    leftEnds_diff=sum(abs(leftEnds-mean(leftEnds)));
    disp(['Eq. no label after LP LeftEnds mean: ' num2str(mean(leftEnds)) ' | LeftEnds difference: ' num2str(leftEnds_diff)]);
    
    current_obj=sum(y_new);
    
    disp(['obj ' num2str(current_obj)]);
    tol=norm(current_obj-initial_obj);
    initial_obj=current_obj;
    loop_i=loop_i+1;
end

fv_H_0=randn(n_sample,1);

[fv_H,~] = ...
    lobpcg_fv(...
    fv_H_0,...
    ABL,...
    1e-4,...
    200);

x_val = sign(fv_H(1))*sign(fv_H(1:n_sample));
% x_val = sign(fv_H(1:n_sample));
x_pred=x_val;
err_count = sum(abs(sign(x_val) - label))/2;

end

