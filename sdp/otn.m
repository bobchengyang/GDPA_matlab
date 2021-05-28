function [new_H] = otn(label,b_ind,n_sample,dL,dy,dz,u,alpha)
%OTN 此处显示有关此函数的摘要
%   此处显示详细说明
db = 2*label(b_ind); % training labels x 2
dz_plus_idx = db < 0;
dz_minus_idx = db > 0;
new_H=[-dL zeros(n_sample+1,1);...
       zeros(n_sample+1,1)' 0];
for i=1:n_sample
    new_H(i,i)=new_H(i,i)+dy(i);
end
new_H(dz_minus_idx,n_sample+1)=dz(dz_minus_idx);
new_H(n_sample+1,dz_minus_idx)=dz(dz_minus_idx);
% new_H(n_sample+1,n_sample+1)=0.5*u-sum(dz(dz_minus_idx));
new_H(n_sample+1,n_sample+1)=u-alpha-sum(dz(dz_minus_idx));
new_H(dz_plus_idx,n_sample+2)=dz(dz_plus_idx);
new_H(n_sample+2,dz_plus_idx)=dz(dz_plus_idx);
% new_H(n_sample+2,n_sample+2)=0.5*u-sum(dz(dz_plus_idx));
new_H(n_sample+2,n_sample+2)=u+alpha-sum(dz(dz_plus_idx));
new_H=(new_H+new_H')/2;
end