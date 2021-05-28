function [y,z]=LP_solve_BCD( ...
    n_sample,...
    b_ind,...
    scaled_factors,...
    rho,...
    dL,...
    db,...
    options)

b_ind_logical=zeros(n_sample+1,1);
b_ind_logical(b_ind)=1;
b_ind_logical=logical(b_ind_logical);

LP_A=zeros(4,2); % number of variables: 2: y_i (+/-) and z_i (+/-)
LP_b=zeros(4,1); % number of variables: 2: y_i (+/-) and z_i (+/-)
counter=1;
for i=1:n_sample+1
    if i~=n_sample+1 % non-last row
    if ismember(i,b_ind) 
        LP_A(i,i)=-1;
        LP_A(i,n_sample+1+counter)=scaled_factors(i,end);
        LP_b(i)=-dL(i,i)-sum(abs(scaled_factors(i,2:end-1).*dL(i,2:end-1)))-rho;
        counter=counter+1;
    else
        LP_A(i,i)=-1;
        LP_b(i)=-dL(i,i)-sum(abs(scaled_factors(i,2:end).*dL(i,2:end)))-rho;
    end
    else % last row
        LP_A(i,i)=-1;
        LP_A(i,n_sample+1+1:end)=scaled_factors(i,b_ind);
        LP_b(i)=-dL(i,i)-sum(abs(scaled_factors(i,~b_ind_logical).*dL(i,~b_ind_logical)))-rho;
    end
end
counter=1;
for i=1:n_sample+1
    if i~=n_sample+1 % non-last row
    if ismember(i,b_ind) 
        LP_A(n_sample+1+i,i)=-1;
        LP_A(n_sample+1+i,n_sample+1+counter)=-scaled_factors(i,end);
        LP_b(n_sample+1+i)=-dL(i,i)-sum(abs(scaled_factors(i,2:end-1).*dL(i,2:end-1)))-rho;
        counter=counter+1;
    else
        LP_A(n_sample+1+i,i)=-1;
        LP_b(n_sample+1+i)=-dL(i,i)-sum(abs(scaled_factors(i,2:end).*dL(i,2:end)))-rho;
    end
    else % last row
        LP_A(n_sample+1+i,i)=-1;
        LP_A(n_sample+1+i,n_sample+1+1:end)=-scaled_factors(i,b_ind);
        LP_b(n_sample+1+i)=-dL(i,i)-sum(abs(scaled_factors(i,~b_ind_logical).*dL(i,~b_ind_logical)))-rho;
    end
end

s_k = linprog([ones(1,n_sample+1) db'],...
    LP_A,LP_b,...
    [],[],...
    [],[],options);
y=s_k(1:n_sample+1);
z=s_k(n_sample+1+1:end);

end

