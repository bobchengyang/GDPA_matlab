function [X,t,estimateMatError,absObjectiveError] = solve_cvx(C,A,b,M,N,Xorig,repNumber)
%{
% data generation

% data parameters
M = 100; % numconstraint
N = 250; % dim
r = 3; % mrank

type = 'R'; % C for complex, R for Real

% prob = 'PM'; % Phasemax problem
prob = 'G'; % general problem

if(strcmp(prob,'PM'))
    r = 1; % for Phasemax problem A should be a vector
end

% generating test data
[Xorig,~,b,A,B,~] = data(r, M, N, prob, type);
%}
useCachedResutls = true
tic;
if useCachedResutls
    load(['SDPT3Output_1itr_' num2str(M) num2str(N)],'SDPT3Output');
    X = SDPT3Output{repNumber}.Xest;
else
cvx_begin sdp
    variable X(N,N) symmetric
    X == semidefinite(N);
    minimize(trace(C'*X))
    subject to
    for i = 1:M
        trace(A{i}'*X) == b(i)
    end
cvx_end
end
t=toc;

% displaying error between the estimated SDP matrix and true matrix
disp('error between the estimated SDP matrix and true matrix')
estimateMatError = norm(X - Xorig*Xorig','fro')/norm(Xorig*Xorig','fro');
disp(estimateMatError);

disp('error between estimated objective and true objective function');
absObjectiveError = (trace(C*(X))-trace(C*(Xorig*Xorig')))/trace(C*(Xorig*Xorig'));
disp(absObjectiveError);