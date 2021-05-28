 function [y,z,obj]=LP_core_Nplus2_abs_noselfloop( ...
    n,...
    scaled_M,...
    scaled_factors,...
    rho,...
    db,...
    options,...
    cL,...
    b_ind,...
    u,...
    alpha,...
    dz_ind_plus,...
    dz_ind_minus)

lbi=length(b_ind); % number of known labels

scaled_M_n=scaled_M(1:n,1:n);
scaled_M_n(1:n+1:end)=0;

%% define [c] [P] [d] in Eq. \eqref{eq:LP_std1} in paper
c=[-ones(n+2,1); -db(:); zeros(lbi,1)]; % N+2*M
d=zeros(n+2+2*lbi,1); % N+2+2*M
d(1:n)=diag(cL)-sum(abs(scaled_M_n),2);
%%========================
d(1:n+2)=d(1:n+2)-rho; % ensure PD
%%========================

%% ======define P starts=======
P=zeros(n+2+2*lbi,n+2+2*lbi);
% the first n rows
P(1:n,1:n)=-eye(n);

for lbi_i=1:lbi
    if sign(db(lbi_i))==1
        scalars=abs(scaled_factors(lbi_i,n+1));        
    else
        scalars=abs(scaled_factors(lbi_i,n+2));
    end
P(lbi_i,n+2+lbi+lbi_i)=scalars;    
end
% the next 1 row
P(n+1,n+1)=-1;
P(n+1,n+2+lbi+dz_ind_minus)=abs(scaled_factors(n+1,dz_ind_minus));
% the next 1 row
P(n+2,n+2)=-1;
P(n+2,n+2+lbi+dz_ind_plus)=abs(scaled_factors(n+2,dz_ind_plus));
% the next lbi rows
P(n+2+1:n+2+lbi,n+2+1:n+2+lbi)=eye(lbi);
P(n+2+1:n+2+lbi,n+2+lbi+1:n+2+2*lbi)=-eye(lbi);
% the last lbi rows
P(n+2+lbi+1:n+2+2*lbi,n+2+1:n+2+lbi)=-eye(lbi);
P(n+2+lbi+1:n+2+2*lbi,n+2+lbi+1:n+2+2*lbi)=-eye(lbi);
%% ======define P ends=======

[s_k,obj] = linprog(-c,...
    P,d,...
    [],[],...
    [],[],options);

s_k=s_k(1:n+2+lbi);
y=s_k(1:n+2);
z=s_k(n+2+1:n+2+lbi);
end