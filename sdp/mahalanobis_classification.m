function [x_pred,err] = mahalanobis_classification(label,data_feature,n_feature,initial_label_index)
%=======Mahalanobis classifier starts========
M=eye(n_feature);
[m,X] = ...
    mahalanobis_classifier_variables(...
    data_feature,...
    label,...
    initial_label_index);
z=mahalanobis_classifier(m,M,X);
x_pred=label;
x_pred(~initial_label_index)=z;
err = sum(abs(sign(x_pred) - label))/2;
%========Mahalanobis classifier ends=========
end

