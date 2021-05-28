function [z]=LP_solve_offdia( ...
    n_sample,...
    scaled_M,...
    scaled_factors,...
    rho,...
    i,...
    db,...
    options)

LP_A=zeros(2,1);
LP_b=zeros(2,1);
LP_A(1)=-scaled_factors(i,n_sample+1);
LP_A(2)=-scaled_factors(n_sample+1,i);
remaning_idx=1:n_sample+1;
remaning_idx([i n_sample+1])=[];
LP_b(1)=scaled_M(i,i)-sum(abs(scaled_M(i,remaning_idx)))-rho;
remaning_idx=1:n_sample;
remaning_idx(i)=[];
LP_b(2)=scaled_M(n_sample+1,n_sample+1)-sum(abs(scaled_M(n_sample+1,remaning_idx)))-rho;
try
    s_k1 = linprog(1,...
        LP_A,LP_b,...
        [],[],...
        [],[],options);
    if isempty(s_k1)
        s_k1=Inf;
    end
catch
    s_k1=Inf;
end

LP_A=zeros(2,1);
LP_b=zeros(2,1);
LP_A(1)=scaled_factors(i,n_sample+1);
LP_A(2)=scaled_factors(n_sample+1,i);
remaning_idx=1:n_sample+1;
remaning_idx([i n_sample+1])=[];
LP_b(1)=scaled_M(i,i)-sum(abs(scaled_M(i,remaning_idx)))-rho;
remaning_idx=1:n_sample;
remaning_idx(i)=[];
LP_b(2)=scaled_M(n_sample+1,n_sample+1)-sum(abs(scaled_M(n_sample+1,remaning_idx)))-rho;
try
    s_k2 = linprog(db,...
        LP_A,LP_b,...
        [],[],...
        [],[],options);
    if isempty(s_k2)
        s_k2=Inf;
    end
catch
    s_k2=Inf;
end

z=min([s_k1 s_k2]);

end

