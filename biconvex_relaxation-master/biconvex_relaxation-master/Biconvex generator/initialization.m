function [X0] = initialization(C,A,b,r,M,N,method,use_obj)

if(strcmp(use_obj,'Y'))
    Uinv = pinv(C);
else
    Uinv = eye(size(C));
end

Yini = zeros(size(C));
ncons = zeros(M,1);

for i = 1:M
    temp = Uinv'*A{i}*Uinv;
    ncons(i) = trace(temp);
    Yini = Yini + b(i)*temp;
end
Yini = Yini/M;
[V,D] = eigs(Yini,r);

if(strcmp(method,'W'))
    % wirtinger flow initialization
    lambda = sqrt(N*sum(b)/sum(ncons));
    X0 = V/norm(V,'fro')*lambda;
else
    % Initialization using "Gradient descent for rank minimization" algorithm
    X0 = V*sqrt(abs(D)/2);    
end

X0 = Uinv*X0;
end