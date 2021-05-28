function [y]=LP_solve_dia( ...
    n_sample,...
    scaled_M,...
    rho,...
    dL,...
    i,...
    options,...
    y,...
    db,...
    z)
remaning_idx=1:n_sample+1;
remaning_idx(i)=[];
LP_A=-1;
LP_b=-dL(i,i)-sum(abs(scaled_M(i,remaning_idx)))-rho;
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
LP_A=1;
LP_b=-dL(i,i)-sum(abs(scaled_M(i,remaning_idx)))-rho;
try
    s_k2 = linprog(1,...
        LP_A,LP_b,...
        [],[],...
        [],[],options);
    if isempty(s_k2)
        s_k2=Inf;
    end
catch
    s_k2=Inf;
end
y(i)=s_k1;
obj_1=ones(1,n_sample+1)*y + db'*z;
y(i)=s_k2;
obj_2=ones(1,n_sample+1)*y + db'*z;
obj=[obj_1 obj_2];
min_idx=obj==min(obj);
s_k=[s_k1 s_k2];
y=s_k(min_idx);
end

