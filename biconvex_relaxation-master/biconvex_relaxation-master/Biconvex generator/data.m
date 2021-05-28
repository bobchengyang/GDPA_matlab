function [X,R,b,A,B,C] = data(r, M, N, L, prob, type, lteq, gteq, eq)
if(strcmp(type,'C'))
    X = randn([N,r]) + randn([N,r])*i;
elseif(strcmp(type,'R'))
    X = randn([N,r]);
else
    X = 2*(randn([N,r])>0)-1;
end
Y = X*X';

% change L to 1 to generate phasemax problem
if(strcmp(prob,'PM'))
    L = ones(M,1);
end

% if the input L is single element then it indicates all constraints are of
% same rank
if(length(L) == 1)
    L = L*ones(M,1);
end

% creating linear objective function
R = cell(M,1);
A = cell(M,1);
b = zeros(M,1);
if(strcmp(type,'C'))
    C = triu(randn(N,N),0) + triu(randn(N,N),0)*i;
else
    C = triu(randn(N,N),0);
end
B = C'*C;

for j = 1:M
    if(strcmp(type,'C'))
        R{j} = triu(randn(L(j),N),0) + triu(randn(L(j),N),0)*i;
    else
        R{j} = triu(randn(L(j),N),0);
    end
    A{j} = R{j}'*R{j};
    
    % to create GOE
    %     A = triu(randn(N),1);
    %     A = A + A' + 2*diag(randn([N,1]));
    %     R{j} = sqrtm(A);
    
    b(j) = norm(R{j}*X,'fro')^2;
%     b(j) = trace(A{j}*Y);
end

% jittering the magnitude of equality constraints to get inequality
% constraints
b(eq+1:eq+lteq) = (1 + rand(lteq,1)).*b(eq+1:eq+lteq);
b(eq+lteq+1:end) = (1 - rand(gteq,1)).*b(eq+lteq+1:end);

end