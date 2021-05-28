function [X, obj, obj_ph, t] = BCR(C,R,b,numiter,beta,gamma,method,N,alpha,flag)
% [X, obj, obj_ph, t] = phasemax_imseg(A,B,b,1000,2,2,'G',m,5,false);
%  % phasemax parameters
%  beta = 2*norm(C,'fro');
%  gamma = beta/2; % anything between 0.3 and 0.5 works best
% clear all;load('temp4.mat');
% Input
% C     : matrix used in objective for image seg x'Cx
% R     : vector or matrix consisting every constraints. Each row vector
%       correspond to single constraint in the formulation
% b     : vector of value bounding each constraint
% beta  : weight parameter, beta > 1
% gamma : weighting between objective function and phasemax function
% alpha : variable for x^Te = alpha
% method: whether the constraints are greater than or less than, 'G','L'
% N     : Number of vertices
% numiter: number of iteration
% Output
% X     : Solution in real number
% obj   : objective value of image segmentation problem
% obj_ph: objective value of phasemax form of problem. This should decrease
% t     : recorded time of every iteration

% initializing objective and time t
obj = zeros(numiter,1);
obj_ph = zeros(numiter,1);
t = zeros(numiter,1);

% creating representation for phasemax
R=cell2mat(R);
M = size(R,1);

if(flag)
    y = [ones(N,1);alpha;b];
    Q1 = [zeros(N+1,1);zeros(M,1)];
    R1 = [eye(N);(ones(1,N)/N);R];
    M1 = N+1;
else
    y = [ones(N,1);b];
    Q1 = [zeros(N,1);zeros(M,1)];
    R1 = [eye(N);R];
    M1 = N;
end
Ainv = (beta*gamma*(R1')*R1 + 2*C')^-1;

% finding initial point for X0 using wirtinger flow alorithm
% step 1 of wirtinger flow initialization
lambda = sqrt(N*sum(y.^2)/norm(R1,'fro')^2);

% step 2 of wirtinger flow initialization
Yini = (R1')*diag(y.^2)*R1/(M+M1);
[V,D] = eigs(Yini,1);
X0 = V/norm(V,2)*lambda;

% Initialization using "Gradient descent for rank minimization" algorithm
X0 = V*sqrt(abs(D)/2);

X = X0;

tic;
for iter = 1:numiter
    % section 1: Compute optimal Q
    % solving for Q using LS
    Q1 = R1*X;
    Q1(1:N) = beta*Q1(1:N)/(beta-1);
    % projecting back Q onto frobenius ball
    if(strcmp(method,'G'))
        Q1(1:M1) = min(abs(Q1(1:M1)),y(1:M1)).*sign(Q1(1:M1));
        Q1(M1+1:end) = max(abs(Q1(M1+1:end)),y(M1+1:end)).*sign(Q1(M1+1:end));
    else
        Q1 = min(abs(Q1),y).*sign(Q1);
    end

   % least square solver for X
   X = Ainv* beta * gamma * R1'*Q1;

    % objective function value
    obj(iter) = X'*C*X;
    obj_ph(iter) = obj(iter) + gamma*(beta*0.5*norm(Q1 - R1*X,2)^2 - 0.5*norm(Q1(1:N),2)^2);
    t(iter) = toc;
end
toc;
end