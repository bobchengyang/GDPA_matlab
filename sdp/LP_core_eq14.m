function [y,z,obj]=LP_core_eq14( ...
    n,...
    scaled_M,...
    scaled_factors,...
    rho,...
    db,...
    options,...
    dL,...
    b_ind)

lbi=length(b_ind); % number of known labels

dL_n=dL(1:n,1:n);
scaled_M_n=scaled_M(1:n,1:n);
scaled_M_n(1:n+1:end)=0;

%% define [c] [P] [d] in Eq. \eqref{eq:LP_std1} in paper
c=[-ones(n+1,1); -db(:); zeros(lbi,1)];
d=zeros(n+1+2*lbi,1);
d(1:n)=diag(dL_n)-sum(abs(scaled_M_n),2);

%%========================
d(1:n+1)=d(1:n+1)-rho; % ensure PD
%%========================

%% ======define P starts=======
P=zeros(n+1+2*lbi);
% the first n rows
P(1:n,1:n+1)=-eye(n,n+1);

for lbi_i=1:lbi
P(b_ind(lbi_i),n+1+lbi+lbi_i)=abs(scaled_factors(b_ind(lbi_i),end));    
end
% P(b_ind',end)=abs(scaled_factors(b_ind',end));
% the next 1 row
P(n+1,n+1)=-1;
P(n+1,n+1+lbi+1:end)=abs(scaled_factors(end,b_ind));
% the next M rows
P(n+1+1:n+1+lbi,n+1+1:n+1+lbi)=eye(lbi);
P(n+1+1:n+1+lbi,n+1+lbi+1:end)=-eye(lbi);
% the last M rows
P(n+1+lbi+1:end,n+1+1:n+1+lbi)=-eye(lbi);
P(n+1+lbi+1:end,n+1+lbi+1:end)=-eye(lbi);
%% ======define P ends=======

[s_k,obj] = linprog(-c,...
    P,d,...
    [],[],...
    [],[],options);

s_k=s_k(1:n+1+lbi);
y=s_k(1:n+1);
z=s_k(n+1+1:end);
end

