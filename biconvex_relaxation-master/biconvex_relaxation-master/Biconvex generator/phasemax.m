function [X, obj, obj_ph, err, t] = phasemax(C, A, R,b,numiter,M,X0,beta,gamma,method, Xorig,lteq, gteq, eq)

if(strcmp(method,'LS'))
    Ainv = (beta*sum(cat(3,A{:}),3) + 2*C')\eye(size(C));
else
    % step size for Gradient descent
    mu = 1/norm(sum(cat(3,A{:}),3),'fro'); % 1e-6
end

obj = zeros(numiter,1);
obj_ph = zeros(numiter,1);
err = zeros(numiter,1);
t = zeros(numiter,1);
Q = cell(M,1);
T = cell(M,1);

X = X0;
tic;    
for iter = 1:numiter
    % section 1: Compute optimal Q; overall it takes 0.13 s for 1000 constraints

    for i = 1:eq
        % solving for Q
        T{i} = R{i}*X;
        Q{i} = beta*T{i}/(beta-gamma);
        
        % projecting back Q onto frobenius ball        
        % using trace norm directly - 3x faster
        nrm = norm(Q{i}(:),2);
        Q{i} = Q{i}*(nrm > sqrt(b(i))) * sqrt(b(i))/nrm + Q{i}*(nrm <= sqrt(b(i)));
    end
    for i = eq+1:eq+lteq
        T{i} = R{i}*X;
        Q{i} = T{i};

        nrm = norm(Q{i}(:),2);
        Q{i} = Q{i}*(nrm > sqrt(b(i))) * sqrt(b(i))/nrm + Q{i}*(nrm <= sqrt(b(i)));        
    end
    for i = M-gteq+1:M
        T{i} = R{i}*X;
        Q{i} = T{i};

        nrm = norm(Q{i}(:),2);
        Q{i} = Q{i}*(nrm < sqrt(b(i))) * sqrt(b(i))/nrm + Q{i}*(nrm >= sqrt(b(i)));        
    end

    if(strcmp(method,'LS'))
        % least square solver for X
        % section 2: matrix multiplication required for least square solver;
        % overall it takes 0.11 s for 1000 constraints
        temp2 = cellfun(@(X,Y) X'*Y, R, Q, 'UniformOutput', false);

        % section 3: take matrix inverse for LS
        X = Ainv * beta * sum(cat(3,temp2{:}),3);
    else
        % Gradient Descent for X; overall it takes 0.11 s
        temp1 = cellfun(@(X,Y,Z) X'*(Y - Z), R, Q, T, 'UniformOutput', false);
        X = X + mu*sum(cat(3,temp1{:}),3);
    end

    % objective function value
    obj(iter) = abs(trace(C*(X*X')));
    obj_ph(iter) = obj(iter);
    for i = 1:eq
        obj_ph(iter) = obj_ph(iter) + beta*0.5*norm(R{i}*X - Q{i},'fro')^2 - 0.5*gamma*norm(Q{i},'fro')^2;
    end
    for i = eq+1:M
        obj_ph(iter) = obj_ph(iter) + beta*0.5*norm(R{i}*X - Q{i},'fro')^2;
    end
    err(iter) = norm(X*X' - Xorig*Xorig','fro')/norm(Xorig*Xorig','fro');
    t(iter) = toc;
    if(iter > 50 && (abs(obj(iter)-obj(iter-1))/obj(iter-1)) <1e-8)
        iter
        err(iter+1:end) = [];
        obj(iter+1:end) = [];
        obj_ph(iter+1:end) = [];
        t(iter+1:end) = [];
        break;
    end
end
toc;

% displaying error between the estimated SDP matrix and true matrix
disp('error between the estimated SDP matrix and true matrix')
disp(norm(X*X' - Xorig*Xorig','fro')/norm(Xorig*Xorig','fro'));

disp('error between estimated objective and true objective function');
disp((abs(trace(C*(X*X')))-abs(trace(C*(Xorig*Xorig'))))/abs(trace(C*(Xorig*Xorig'))));

% plotting phasemax objective function; original objective function and
% error between the original SDP matrix and estimated matrix
figure; 

subplot(3,1,1);
plot(obj_ph,'k');
title('Biconvex Objective function');

subplot(3,1,2);
plot(obj,'k');
title('Objective function of original problem');

subplot(3,1,3);
plot(err,'k');
title('error between original and estimated SDP matrix');
end