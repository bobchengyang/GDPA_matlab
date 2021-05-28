 function [y,z,obj]=LP_core_Nplus2( ...
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
    sw,...
    dz_ind_plus,...
    dz_ind_minus)

lbi=length(b_ind); % number of known labels

scaled_M_n=scaled_M(1:n,1:n);
scaled_M_n(1:n+1:end)=0;

%% define [c] [P] [d] in Eq. \eqref{eq:LP_std1} in paper
c=[ones(n+1,1); db(:)]; % N+M
d=zeros(n+2+lbi,1); % N+2+M
d(1:n)=diag(cL)-sum(abs(scaled_M_n),2);
d(n+1)=-alpha;
d(n+2)=alpha;
%%========================
d(1:n+2)=d(1:n+2)-rho; % ensure PD
%%========================

%% ======define P starts=======
P=zeros(n+2+lbi,n+1+lbi);
% the first n rows
P(1:n,1:n)=-eye(n);

for lbi_i=1:lbi
    if sign(db(lbi_i))==1 % b_i>0, i.e., z_i<0
        scalars=abs(scaled_factors(lbi_i,n+1));
        P(lbi_i,n+1+lbi_i)=-scalars;    
        % the last lbi rows
        P(n+2+lbi_i,n+1+lbi_i)=1;  
        % the n+1 th row
        P(n+1,n+1+lbi_i)=-sw+1-abs(scaled_factors(n+1,lbi_i));  
        % the n+2 th row
        P(n+2,n+1+lbi_i)=-(1-sw);  
    else                  % b_i<0, i.e., z_i>0
        scalars=abs(scaled_factors(lbi_i,n+2));
        P(lbi_i,n+1+lbi_i)=scalars;   
        % the last lbi rows
        P(n+2+lbi_i,n+1+lbi_i)=-1;  
        % the n+1 th row
        P(n+1,n+1+lbi_i)=-sw;  
        % the n+2 th row
        P(n+2,n+1+lbi_i)=-(1-sw)+1+abs(scaled_factors(n+2,lbi_i));     
    end
end
% the n+1 th row
P(n+1,n+1)=-sw;
% the n+2 th row
P(n+2,n+1)=-(1-sw);
%% ======define P ends=======

[s_k,obj] = linprog(c,...
    P,d,...
    [],[],...
    [],[],options);

y=s_k(1:n+1);
z=s_k(n+1+1:n+1+lbi);
end