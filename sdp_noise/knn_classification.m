function [x_pred,err] = knn_classification(label,label_noisy,data_feature,n_feature,initial_label_index)
knn_size = 10;
M=eye(n_feature);
%========KNN classifier starts========
fl = label_noisy(initial_label_index);
fl(fl == -1) = 0;
x = KNN(fl, data_feature(initial_label_index,:), sqrtm(M), knn_size, data_feature(~initial_label_index,:));
x(x==0) = -1;
x_pred = label;
x_pred(~initial_label_index) = x;
err = sum(abs(sign(x) - label(~initial_label_index)))/2;
%=========KNN classifier ends=========
end
