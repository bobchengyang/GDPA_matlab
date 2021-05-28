function [y_new,current_obj,x_pred,err_count] = eq_closed_form_fast(gamma,label,n_sample,cL,b_ind)
%eq_no_label Summary of this function goes here
%   Detailed explanation goes here
db = 2*label(b_ind); % training labels x 2
y_1toN=zeros(n_sample,1)+gamma;
lmin=gamma;
dz=-(lmin/2)*db;
y_np1=sum(dz.^2)/lmin;
current_obj=sum(y_1toN)+y_np1+db'*dz;

y_new=[y_1toN;y_np1];

fv_H_0=randn(n_sample+1,1);

ABL=[cL+diag(y_1toN) [dz;zeros(n_sample-length(b_ind),1)] ;[dz' zeros(n_sample-length(b_ind),1)'] y_np1];

[fv_H,lm] = ...
    lobpcg_fv(...
    fv_H_0,...
    ABL,...
    1e-16,...
    1e3);

if lm<=0
    disp('=================');
    disp('H is NOT PSD!!!!');
    disp('=================');
end

x_val = sign(label(b_ind(1))*sign(fv_H(1)))*sign(fv_H(1:n_sample));
% x_val = sign(fv_H(1:n_sample));
x_pred=x_val;
err_count = sum(abs(sign(x_val) - label))/2;

end

