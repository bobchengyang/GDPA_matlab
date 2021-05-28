 function [y,z,obj]=LP_core_Nplus2_abs_noselfloop_vec( ...
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
% P=zeros(n+2+2*lbi,n+2+2*lbi);
P_sparse_i=zeros(n+2*lbi+2+4*lbi,1);
P_sparse_j=zeros(n+2*lbi+2+4*lbi,1);
P_sparse_s=zeros(n+2*lbi+2+4*lbi,1);

% the first n rows
% P(1:n,1:n)=-eye(n);
P_sparse_i(1:n)=1:n;
P_sparse_j(1:n)=1:n;
P_sparse_s(1:n)=-1;

for lbi_i=1:lbi
    if sign(db(lbi_i))==1
        scalars=abs(scaled_factors(lbi_i,n+1));        
    else
        scalars=abs(scaled_factors(lbi_i,n+2));
    end
% P(lbi_i,n+2+lbi+lbi_i)=scalars;   
P_sparse_i(n+lbi_i)=lbi_i;
P_sparse_j(n+lbi_i)=n+2+lbi+lbi_i;
P_sparse_s(n+lbi_i)=scalars;		 
end
% the next 1 row
% P(n+1,n+1)=-1;

P_sparse_i(n+lbi+1)=n+1;
P_sparse_j(n+lbi+1)=n+1;
P_sparse_s(n+lbi+1)=-1;

% P(n+1,n+2+lbi+dz_ind_minus)=abs(scaled_factors(n+1,dz_ind_minus));

P_sparse_i(n+lbi+1+1:n+lbi+1+length(dz_ind_minus))=n+1;
P_sparse_j(n+lbi+1+1:n+lbi+1+length(dz_ind_minus))=n+2+lbi+dz_ind_minus;
P_sparse_s(n+lbi+1+1:n+lbi+1+length(dz_ind_minus))=abs(scaled_factors(n+1,dz_ind_minus));

% the next 1 row
% P(n+2,n+2)=-1;

P_sparse_i(n+lbi+1+length(dz_ind_minus)+1)=n+2;
P_sparse_j(n+lbi+1+length(dz_ind_minus)+1)=n+2;
P_sparse_s(n+lbi+1+length(dz_ind_minus)+1)=-1;

% P(n+2,n+2+lbi+dz_ind_plus)=abs(scaled_factors(n+2,dz_ind_plus));

P_sparse_i(n+lbi+2+length(dz_ind_minus)+1:n+2*lbi+2)=n+2;
P_sparse_j(n+lbi+2+length(dz_ind_minus)+1:n+2*lbi+2)=n+2+lbi+dz_ind_plus;
P_sparse_s(n+lbi+2+length(dz_ind_minus)+1:n+2*lbi+2)=abs(scaled_factors(n+2,dz_ind_plus));

% the next lbi rows
% P(n+2+1:n+2+lbi,n+2+1:n+2+lbi)=eye(lbi);
% P(n+2+1:n+2+lbi,n+2+lbi+1:n+2+2*lbi)=-eye(lbi);

P_sparse_i(n+2*lbi+2+1:n+2*lbi+2+lbi)=n+2+1:n+2+lbi;
P_sparse_j(n+2*lbi+2+1:n+2*lbi+2+lbi)=n+2+1:n+2+lbi;
P_sparse_s(n+2*lbi+2+1:n+2*lbi+2+lbi)=1;

P_sparse_i(n+2*lbi+2+lbi+1:n+2*lbi+2+2*lbi)=n+2+1:n+2+lbi;
P_sparse_j(n+2*lbi+2+lbi+1:n+2*lbi+2+2*lbi)=n+2+lbi+1:n+2+2*lbi;
P_sparse_s(n+2*lbi+2+lbi+1:n+2*lbi+2+2*lbi)=-1;

% the last lbi rows
% P(n+2+lbi+1:n+2+2*lbi,n+2+1:n+2+lbi)=-eye(lbi);
% P(n+2+lbi+1:n+2+2*lbi,n+2+lbi+1:n+2+2*lbi)=-eye(lbi);

P_sparse_i(n+2*lbi+2+2*lbi+1:n+2*lbi+2+3*lbi)=n+2+lbi+1:n+2+2*lbi;
P_sparse_j(n+2*lbi+2+2*lbi+1:n+2*lbi+2+3*lbi)=n+2+1:n+2+lbi;
P_sparse_s(n+2*lbi+2+2*lbi+1:n+2*lbi+2+3*lbi)=-1;

P_sparse_i(n+2*lbi+2+3*lbi+1:n+2*lbi+2+4*lbi)=n+2+lbi+1:n+2+2*lbi;
P_sparse_j(n+2*lbi+2+3*lbi+1:n+2*lbi+2+4*lbi)=n+2+lbi+1:n+2+2*lbi;
P_sparse_s(n+2*lbi+2+3*lbi+1:n+2*lbi+2+4*lbi)=-1;

P=sparse(P_sparse_i,P_sparse_j,P_sparse_s,n+2+2*lbi,n+2+2*lbi);
%% ======define P ends=======

[s_k,obj] = linprog(-c,...
    P,d,...
    [],[],...
    [],[],options);

s_k=s_k(1:n+2+lbi);
y=s_k(1:n+2);
z=s_k(n+2+1:n+2+lbi);
end