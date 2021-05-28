function [ s_k ] = ...
    LP_solve_1offdia_2dia_core( ...
    sign_vec,...
    options,...
    db,...
    rho,...
    scaled_M,...
    scaled_factors,...
    n_sample,...
    i,...
    y,...
    y_sign)

LP_A=zeros(2,3);
LP_b=zeros(2,1);
LP_A(1,2)=-1;
LP_A(2,3)=-1;
LP_A(1,1)=sign_vec*sign(scaled_factors(i,n_sample+1))*abs(scaled_factors(i,n_sample+1));
LP_A(2,1)=sign_vec*sign(scaled_factors(n_sample+1,i))*abs(scaled_factors(n_sample+1,i));
remaning_idx=1:n_sample+1;
remaning_idx([i n_sample+1])=[];
LP_b(1)=scaled_M(i,i)-y(i)-sum(abs(scaled_M(i,remaning_idx)))-rho;
remaning_idx=1:n_sample;
remaning_idx(i)=[];
LP_b(2)=scaled_M(n_sample+1,n_sample+1)-y(n_sample+1)-sum(abs(scaled_M(n_sample+1,remaning_idx)))-rho;
lb=zeros(3,1);
ub=zeros(3,1);
if y_sign(i)==1
    lb(2)=0;
    ub(2)=Inf;
else
    lb(2)=-Inf;
    ub(2)=0;
end
if y_sign(n_sample+1)==1
    lb(3)=0;
    ub(3)=Inf;
else
    lb(3)=-Inf;
    ub(3)=0;
end
if sign_vec==1
    lb(1)=0;
    ub(1)=Inf;
else
    lb(1)=-Inf;
    ub(1)=0;
end
try
    s_k = linprog([db(1) 1 1],...
        LP_A,LP_b,...
        [],[],...
        lb,ub,options);
    if isempty(s_k)
        s_k=Inf;
    end
catch
    s_k=Inf;
end
end

