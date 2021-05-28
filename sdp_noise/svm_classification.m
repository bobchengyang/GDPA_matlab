function [x_pred,err] = svm_classification(label,label_noisy,data_feature,initial_label_index)
SVMModel = fitcsvm(data_feature(initial_label_index,:),label_noisy(initial_label_index),'KernelScale','auto','Standardize',false,...
    'OutlierFraction',0.00);
[x,~,~] = predict(SVMModel,data_feature(~initial_label_index,:));
x_pred=label;
x_pred(~initial_label_index)=x;
err = sum(abs(sign(x) - label(~initial_label_index)))/2;
end

