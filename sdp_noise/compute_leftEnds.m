function [s,leftEnds] = compute_leftEnds(M)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
[V,Lamb] = eig(M);
v = V(:,1);
N = length(v);
s = 1.0./v;

for i=1:N,
    leftEnds(i) = M(i,i);
    for j=1:N,
        if j ~= i,
            leftEnds(i) = leftEnds(i) - abs(s(i)/s(j)*M(i,j));
        end
    end
end

end

