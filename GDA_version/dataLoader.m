function [A,label] = dataLoader(path)
%DATALOADER Summary of this function goes here
%   Detailed explanation goes here
  function [val] = cal_function(f1,f2)
    val = exp(-sum((f1-f2).*(f1-f2)));
  end
data = csvread(path);
[num,dim] = size(data);
A = zeros(num,num);
for i=1:num
  for j=1:num
      if i~=j
    A(i,j) = cal_function(data(i,1:dim-1),data(j,1:dim-1));
      end
  end
end
label = data(:,end);
label(label==2)=-1;
end
