function [x_pred,error_count,obj] = GLR_closed_form(...
    L,...
    n_sample,...
    b_ind,...
    class_train_test)
%GLR_CLOSED_FORM Summary of this function goes here
%   Detailed explanation goes here
initial_label_index=zeros(n_sample,1);
initial_label_index(b_ind)=1;
initial_label_index=logical(initial_label_index);
x_pred=-pinv(L(~initial_label_index,~initial_label_index))...
    *L(~initial_label_index,initial_label_index)...
    *class_train_test(initial_label_index,:);
x_valid=class_train_test;
x_valid(~initial_label_index)=x_pred;
error_count = sum(abs(sign(x_valid) - class_train_test))/2;
x_valid=sign(x_valid);
obj=x_valid'*L*x_valid;
end

