function [X, obj, obj_ph, err, t,estimateMatError,absObjectiveError] = phasemax_vector(C, A, R,b,L,...
    numiter,M,X0,beta,gamma,method, Xorig,lteq, gteq, eq)

dontPlotPhaseMaxOutput = false; %Set dontPlotPhaseMaxOutput to false to print all the phasemax and error plots

if(strcmp(method,'LS'))
    Ainv = (beta*sum(cat(3,A{:}),3) + 2*C')\eye(size(C));
else
    % step size for Gradient descent
    mu = 1/norm(sum(cat(3,A{:}),3),'fro'); % 1e-6
end

% taking square root of magnitude of constraints
b = sqrt(b);
cons_ind = cumsum(L);
R = cat(1,R{:});

obj = zeros(numiter,1);
obj_ph = zeros(numiter,1);
err = zeros(numiter,1);
t = zeros(numiter,1);

X = X0;
tic;
for iter = 1:numiter
    Xold = X;
    % section 1: Compute optimal Q;
    T = R*X;
    Q = T;
    Q(1:sum(L(1:eq)),:) = (beta/(beta-gamma))*Q(1:sum(L(1:eq)),:);
    
    % normalizing: projecting within the ball
    nrm_temp = cumsum(sum(Q.^2,2));
    nrm = sqrt(diff([0;nrm_temp(cons_ind)]));
    
    % min for projection within ball; max for outside the ball
    normalization = [min(b(1:eq+lteq),nrm(1:eq+lteq));max(b(eq+lteq+1:end),nrm(eq+lteq+1:end))]./(nrm+eps);
    
    %%%%%%%% this code runs slower for large matrices in the constraint %%%%%%%
    %     normalization_expand = zeros(cons_ind(end),1);
    %     normalization_expand(1:L(1)) = normalization(1)*ones(L(1),1);
    %     for i = 2:M
    %         normalization_expand(cons_ind(i-1)+1:cons_ind(i)) = normalization(i)*ones(L(i),1);
    %     end
    %     Q = spdiags(normalization_expand,0,cons_ind(end),cons_ind(end))*Q;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    Q(1:cons_ind(1),:) = normalization(1)*Q(1:cons_ind(1),:);
    for i = 2:M
        Q(cons_ind(i-1)+1:cons_ind(i),:) = normalization(i)*Q(cons_ind(i-1)+1:cons_ind(i),:);
    end
    
    if(strcmp(method,'LS'))
        % least square solver for X
        % section 2: matrix multiplication required for least square solver;
        % section 3: take matrix inverse for LS
        
        X = beta * Ainv * (R'*Q);
    else
        % Gradient Descent for X; overall it takes 0.11 s
        X = X + mu*R'*(Q-T);
    end
    
    % objective function value
    %Set dontPlotPhaseMaxOutput to false to print all the phasemax and error plots
    if dontPlotPhaseMaxOutput
        obj(iter) = abs(trace(C*(X*X')));
        obj_ph(iter) = obj(iter);
        obj_ph(iter) = obj_ph(iter) + beta*0.5*norm(R*X - Q,'fro')^2 - 0.5*gamma*norm(Q(1:sum(L(1:eq)),:),'fro')^2;
        err(iter) = norm(X*X' - Xorig*Xorig','fro')/norm(Xorig*Xorig','fro');
    end
    t(iter) = toc;
    if(iter > 50 && norm(X-Xold,'fro') <1e-8)
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
estimateMatError =norm(X*X' - Xorig*Xorig','fro')/norm(Xorig*Xorig','fro');
disp(estimateMatError);

disp('error between estimated objective and true objective function');
absObjectiveError =(abs(trace(C*(X*X')))-abs(trace(C*(Xorig*Xorig'))))/abs(trace(C*(Xorig*Xorig')));
disp(absObjectiveError);

% plotting phasemax objective function; original objective function and
% error between the original SDP matrix and estimated matrix

%Set dontPlotPhaseMaxOutput to false to print all the phasemax and error plots
if dontPlotPhaseMaxOutput
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

end