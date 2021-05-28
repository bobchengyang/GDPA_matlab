function [L,feature,n_feature,label] = dataLoader_normalized(data,n_sample,rsrng)
%DATALOADER Summary of this function goes here
%   Detailed explanation goes here
%   function [val] = cal_function(f1,f2)
%     val = exp(-sum((f1-f2).*(f1-f2)));
%   end
%data = csvread(path);
rng(rsrng); % random seed 0
reorderx=randperm(n_sample);
reorderx=reorderx(1:n_sample);
% K=round(num/n_sample); % test 4 samples at a time

% data_idx=(indices==1);
selected_data=data(reorderx,:);

feature=selected_data(:,1:end-1);

feature(isnan(feature))=0;

n_feature=size(feature,2);

label=selected_data(:,end);
label(label~=1)=-1;

mean_set_0 = mean(feature);
std_set_0 = std(feature);

mean_set = repmat(mean_set_0,size(feature,1),1);
std_set = repmat(std_set_0,size(feature,1),1);

feature = (feature - mean_set)./std_set;

if length(find(isnan(feature)))>0
    error('features have NaN(s)');
end

feature_l2=sqrt(sum(feature.^2,2));
for i=1:size(feature,1)
    feature(i,:)=feature(i,:)/feature_l2(i);
end

[c,y] = get_graph_Laplacian_variables_ready(feature,label,n_sample,n_feature);
M=eye(n_feature);
[L] = graph_Laplacian( n_sample, c, M );
   
% A = zeros(num,num);
% for i=1:num
%   for j=1:num
%       if i~=j
%     A(i,j) = cal_function(data(i,1:dim-1),data(j,1:dim-1));
%       end
%   end
% end
% label = data(:,end);
% label(label==2)=-1;
end
