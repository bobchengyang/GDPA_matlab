 function [y,z,obj]=LP_core_Nplus2_scalars( ...
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
c=[ones(n+1,1); db(:); zeros(lbi,1)]; % N+1+2*M
d=zeros(n+2+2*lbi,1); % N+2+M
d(1:n)=diag(cL)-sum(abs(scaled_M_n),2);
d(n+1)=-alpha;
d(n+2)=alpha;
%%========================
d(1:n+2)=d(1:n+2)-rho; % ensure PD
%%========================

%% ======define P starts=======
P=zeros(n+2+2*lbi,n+1+2*lbi);
% the first n rows
P(1:n,1:n)=-eye(n);

for lbi_i=1:lbi
    if sign(db(lbi_i))==1 % b_i>0, i.e., z_i<0
        scalars=abs(scaled_factors(lbi_i,n+1));   
        % the n+1 th row
        P(n+1,n+1+lbi_i)=-sw+1;  % X_{1xM}
        P(n+1,n+1+lbi+lbi_i)=abs(scaled_factors(n+1,lbi_i));  % X'_{1xM}
        % the n+2 th row
        P(n+2,n+1+lbi_i)=-(1-sw);  
    else                  % b_i<0, i.e., z_i>0
        scalars=abs(scaled_factors(lbi_i,n+2));
        % the n+1 th row
        P(n+1,n+1+lbi_i)=-sw;  
        % the n+2 th row
        P(n+2,n+1+lbi_i)=-(1-sw)+1; % Y_{1xM}   
        P(n+2,n+1+lbi+lbi_i)=abs(scaled_factors(n+2,lbi_i)); % Y'_{1xM}
    end
    P(lbi_i,n+1+lbi+lbi_i)=scalars; % E_{NxM}
end
% the n+1 th row
P(n+1,n+1)=-sw;
% the n+2 th row
P(n+2,n+1)=-(1-sw);
% the next lbi rows
P(n+2+1:n+2+lbi,n+1+1:n+1+lbi)=eye(lbi);
P(n+2+1:n+2+lbi,n+1+lbi+1:n+1+2*lbi)=-eye(lbi);
% the last lbi rows
P(n+2+lbi+1:n+2+2*lbi,n+1+1:n+1+lbi)=-eye(lbi);
P(n+2+lbi+1:n+2+2*lbi,n+1+lbi+1:n+1+2*lbi)=-eye(lbi);
%% ======define P ends=======

% Aeq=zeros(lbi-2,n+1+2*lbi);
% beq=zeros(lbi-2,1);
% for i=1:length(dz_ind_minus)-1
%     Aeq(i,n+1+dz_ind_minus(1))=1;
%     Aeq(i,n+1+dz_ind_minus(1+i))=-1;
% end
% for i=1:length(dz_ind_plus)-1
%     Aeq(length(dz_ind_minus)-1+i,n+1+dz_ind_plus(1))=1;
%     Aeq(length(dz_ind_minus)-1+i,n+1+dz_ind_plus(1+i))=-1;
% end
[s_k,obj] = linprog(c,...
    P,d,...
    [],[],...
    [],[],options);

s_k=s_k(1:n+1+lbi);
y=s_k(1:n+1);
z=s_k(n+1+1:n+1+lbi);
end