function [m,X] = ...
    mahalanobis_classifier_variables(...
    feature_train_test,...
    class_train_test,...
    initial_label_index)

% feature_train_class_1=feature_train_test(class_train_test==1,:);
% feature_train_class_2=feature_train_test(class_train_test==-1,:);

feature_train=feature_train_test(initial_label_index,:);
class_train=class_train_test(initial_label_index);
feature_train_class_1=feature_train(class_train==1,:);
feature_train_class_2=feature_train(class_train==-1,:);

if size(feature_train_class_1,1)>1
    mf1=mean(feature_train_class_1);
else
    mf1=feature_train_class_1;
end
if size(feature_train_class_2,1)>1
    mf2=mean(feature_train_class_2);
else
    mf2=feature_train_class_2;
end

m=[mf1;mf2];

X=feature_train_test(~initial_label_index,:);

m=m';
X=X';

end

