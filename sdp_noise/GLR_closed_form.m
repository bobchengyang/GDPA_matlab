function [x_pred,error_count,obj,t_orig_end] = GLR_closed_form(...
    L,...
    n_sample,...
    b_ind,...
    class_train_test,...
    label_noisy)
%GLR_CLOSED_FORM Summary of this function goes here
%   Detailed explanation goes here
initial_label_index=zeros(n_sample,1);
initial_label_index(b_ind)=1;
initial_label_index=logical(initial_label_index);
t_orig=tic;
x_pred=-pinv(L(~initial_label_index,~initial_label_index))...
    *L(~initial_label_index,initial_label_index)...
    *label_noisy(initial_label_index,:);
t_orig_end=toc(t_orig);
x_pred=sign(x_pred);
x_valid=class_train_test;
x_valid(~initial_label_index)=x_pred;
% error_count = sum(abs(sign(x_valid) - class_train_test))/2;
error_count = sum(abs(sign(x_pred) - class_train_test(~initial_label_index)))/2;
x_valid=sign(x_valid);
obj=x_valid'*L*x_valid;

% cvx_begin
% variable x(n_sample,1);
% minimize(x'*L*x)
% subject to
% x(initial_label_index) == class_train_test(initial_label_index);
% cvx_end
% 
% x_valid = sign(x);
% error_count = sum(abs(sign(x_valid) - class_train_test))/2;
% obj=x_valid'*L*x_valid;
end

