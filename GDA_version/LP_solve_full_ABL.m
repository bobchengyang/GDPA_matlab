function [y,z,obj]=LP_solve_full_ABL( ...
    n,...
    scaled_M,...
    scaled_factors,...
    rho,...
    db,...
    options,...
    dL,...
    b_ind)

lbi=length(b_ind); % number of known labels
nov=2*(n+1)+2*lbi; % number of variables (original y's z's + absolute values of y's z's)
LP_A=zeros(n+1+nov,nov);
LP_b=zeros(n+1+nov,1);

lb=zeros(nov,1);
ub=zeros(nov,1);
lb(1:n+1+lbi)=-Inf; % y's z's can be +/-
ub(1:end)=Inf; % abs(y)'s abs(z)'s >= 0

counter=0;
add_row=0;

%% ========keep in mind=========
% 1. define additional variables for variables with absolute values
%% =============================

for cl=1:n
    if ismember(cl,b_ind) % both y and z exist
        counter=counter+1; 
               
        % y_i+L_{ii} - abs(\sum_{j \neq i}{s_i/s_j*M_ij}) >= rho
        % -y_i + abs(scaled_factors(cl,end))*abs(z_i) <=
        % L_{ii}-abs(\sum_{j \neq i}{s_i/s_j*M_ij}) -rho
        LP_A(cl,cl)=-1; % y factor
        LP_A(cl,n+1+lbi+n+1+counter)=...
            abs(scaled_factors(cl,end)); % z factor
        remaning_idx=1:n+1;
        remaning_idx([cl n+1])=[]; % remove i and n+1
        LP_b(cl)=dL(cl,cl)...
            -sum(abs(scaled_M(cl,remaning_idx)))...
            -rho;
        
        % y_i - abs(y_i) <= 0
        % -y_i - abs(y_i) <= 0
        % z_i - abs(z_i) <= 0
        % -z_i - abs(z_i) <= 0
        add_row=add_row+1;
        LP_A(n+1+add_row,cl)=1;
        LP_A(n+1+add_row,cl+n+1+lbi)=-1;
        add_row=add_row+1;
        LP_A(n+1+add_row,cl)=-1;
        LP_A(n+1+add_row,cl+n+1+lbi)=-1;
        add_row=add_row+1;
        LP_A(n+1+add_row,n+1+counter)=1;
        LP_A(n+1+add_row,n+1+counter+n+1+lbi)=-1;
        add_row=add_row+1;
        LP_A(n+1+add_row,n+1+counter)=-1;
        LP_A(n+1+add_row,n+1+counter+n+1+lbi)=-1;
        
    else % only y exists
 
        % y_i+L_{ii} - abs(\sum_{j \neq i}{s_i/s_j*M_ij}) >= rho
        % -y_i <= L_{ii}-abs(\sum_{j \neq i}{s_i/s_j*M_ij}) -rho
        
        LP_A(cl,cl)=-1;
%         LP_A(cl,n+1+lbi+n+1+counter)=...
%             abs(scaled_factors(cl,end));
        remaning_idx=1:n+1;
        remaning_idx(cl)=[];
        LP_b(cl)=dL(cl,cl)...
            -sum(abs(scaled_M(cl,remaning_idx)))...
            -rho;
   
        % y_i - abs(y_i) <= 0
        % -y_i - abs(y_i) <= 0
        add_row=add_row+1;
        LP_A(n+1+add_row,cl)=1;
        LP_A(n+1+add_row,cl+n+1+lbi)=-1;
        add_row=add_row+1;
        LP_A(n+1+add_row,cl)=-1;
        LP_A(n+1+add_row,cl+n+1+lbi)=-1;
%         LP_A(n+1+4*(cl-1)+3,n+1+counter)=1;
%         LP_A(n+1+4*(cl-1)+3,n+1+counter+n+1+lbi)=-1;
%         LP_A(n+1+4*(cl-1)+4,n+1+counter)=-1;
%         LP_A(n+1+4*(cl-1)+4,n+1+counter+n+1+lbi)=-1;

    end 
end

        % y_i+L_{ii} - abs(\sum_{j \neq i}{s_i/s_j*M_ij}) >= rho
        % -y_i + \sum_{j\neq i}{abs(scaled_M(end,b_ind))} <= L_{ii}-rho
        
        LP_A(n+1,n+1)=-1;
        LP_A(n+1,n+1+lbi+n+1+1:end)=...
            abs(scaled_factors(end,b_ind));
        
%         remaning_idx=1:n+1;
%         remaning_idx(cl)=[];
        LP_b(n+1)=dL(n+1,n+1)-rho;
 
        add_row=add_row+1;
        LP_A(n+1+add_row,n+1)=1;
        LP_A(n+1+add_row,n+1+n+1+lbi)=-1;
        add_row=add_row+1;
        LP_A(n+1+add_row,n+1)=-1;
        LP_A(n+1+add_row,n+1+n+1+lbi)=-1;

% LP_A(1,1)=sign_vec*sign(scaled_factors(i,n+1))*abs(scaled_factors(i,n+1));
% LP_A(2,1)=sign_vec*sign(scaled_factors(n+1,i))*abs(scaled_factors(n+1,i));
% remaning_idx=1:n+1;
% remaning_idx([i n+1])=[];
% LP_b(1)=scaled_M(i,i)-y(i)-sum(abs(scaled_M(i,remaning_idx)))-rho;
% remaning_idx=1:n;
% remaning_idx(i)=[];
% LP_b(2)=scaled_M(n+1,n+1)-y(n+1)-sum(abs(scaled_M(n+1,remaning_idx)))-rho;

try
    [s_k,obj] = linprog([ones(1,n+1) db' zeros(1,n+1+lbi)],...
        LP_A,LP_b,...
        [],[],...
        lb,ub,options);
    if isempty(s_k)
        s_k=Inf;
    end
catch
    s_k=Inf;
end
s_k=s_k(1:n+1+lbi);
y=s_k(1:n+1);
z=s_k(n+1+1:end);
end

