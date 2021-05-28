 function [y,z,obj]=LP_core_Nplus2_abs( ...
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
% db(dz_ind_minus)=db(dz_ind_minus)-1;
c=[-ones(n,1); -db(:); zeros(lbi,1)]; % N+2*M
d=zeros(n+2+2*lbi,1); % N+2+2*M
d(1:n)=diag(cL)-sum(abs(scaled_M_n),2);
d(n+1)=0.5*u(n+1)-alpha;
d(n+2)=0.5*u(n+1)+alpha;
%%========================
d(1:n+2)=d(1:n+2)-rho; % ensure PD
%%========================

%% ======define P starts=======
P=zeros(n+2+2*lbi,n+2*lbi);
% the first n rows
P(1:n,1:n)=-eye(n);

for lbi_i=1:lbi
    if sign(db(lbi_i))==1
        scalars=abs(scaled_factors(lbi_i,n+1));
        P(n+1,n+lbi+lbi_i)=1+abs(scaled_factors(n+1,lbi_i));            
    else
        scalars=abs(scaled_factors(lbi_i,n+2));
        P(n+2,n+lbi+lbi_i)=1+abs(scaled_factors(n+2,lbi_i));    
    end
P(lbi_i,n+lbi+lbi_i)=scalars;    
end
% the next lbi rows
P(n+2+1:n+2+lbi,n+1:n+lbi)=eye(lbi);
P(n+2+1:n+2+lbi,n+lbi+1:end)=-eye(lbi);
% the last lbi rows
P(n+2+lbi+1:n+2+2*lbi,n+1:n+lbi)=-eye(lbi);
P(n+2+lbi+1:n+2+2*lbi,n+lbi+1:end)=-eye(lbi);
%% ======define P ends=======

[s_k,obj] = linprog(-c,...
    P,d,...
    [],[],...
    [],[],options);

s_k=s_k(1:n+lbi);
y=s_k(1:n);
z=s_k(n+1:n+lbi);
obj=obj+(-sum(z(dz_ind_minus))+0.5*u(n+1)-alpha);
end