function [sdpt3Output] = wrapper_general_SDPT3(M,N,repNumber)

%% data generation
type = 'R'; % C for complex, R for Real, B for binary
prob = 'G'; % 'G' general problem, 'PM' Phasemax problem
obj = 'Y'; % indicates whether problem has any objective function or not

% data parameters
if nargin==0
M = 250; % total constraint
N = 250; % dim
end
lteq = 0;
gteq = 0;
eq = M-lteq-gteq;


L = N; % rank of constraint. It can be a vector of length M.
r = 3; % rank of solution X
numiter = 100;%10000;

if(strcmp(prob,'PM'))
    r = 1; 
    obj = 'N';
    L = 1; % for Phasemax problem, each constraint matrix should be a vector
end

% if the rank of constraint 'L' is single element then it indicates all 
% constraints are of same rank
if(length(L) == 1)
    L = L*ones(M,1);
end

% for M = [1000 1500 2000 2500 3000]
% generating test data; the order of constraints is equality, less than,
% greater than
[Xorig,R,b,A,B,C] = data(r, M, N, L, prob, type, lteq, gteq, eq);

% some preprocessing
B = B/max(norm(B,'fro'),eps);
C = C/max(sqrt(norm(B,'fro')),eps);

%% initialization

% initialization parameters
init_method = 'GD';%'GD'; % initialization method, use W for wirtinger flow like, GD for Gradient descent like
use_obj = 'N'; % whether to use linear matrix C from objective for initialization
if(strcmp(obj,'N'))
    use_obj = 'N';
    B = zeros(size(B));
    C = zeros(size(C));
end

X0 = initialization(C,A,b,r,M,N,init_method,use_obj);

disp('Error between original matrix and initial matrix');
initialMatError = norm(X0*X0' - Xorig*Xorig','fro')/norm(Xorig*Xorig','fro');
disp(initialMatError);
sdpt3Output.matError=initialMatError;


%% phasemax algorithm

% phasemax parameters
beta = 2*norm(C,'fro');
gamma = beta/2; % anything between 0.3 and 0.5 works best

[X,t,estimateMatError,absObjectiveError] = solve_cvx(B,A,b,M,N,Xorig,repNumber);
sdpt3Output.t =t;
sdpt3Output.Xest=X;
sdpt3Output.estimateMatError = estimateMatError;
sdpt3Output.absObjectiveError = absObjectiveError;


% using Least square approach
% [Xest, obj, obj_ph, err, t] = phasemax(B,A,R,b, numiter,M,X0,beta,gamma,'LS', Xorig, lteq, gteq, eq);
% [Xest, obj, obj_ph, err, t,estimateMatError,absObjectiveError] = phasemax_vector(B,A,R,b,L, numiter,M,X0,beta,gamma,'LS', Xorig, lteq, gteq, eq);
% phmaxOutput.t =t(end);
% phmaxOutput.iterations = length(t);
% phmaxOutput.Xest=Xest;
% phmaxOutput.estimateMatError = estimateMatError;
% phmaxOutput.absObjectiveError = absObjectiveError;
% % using gradient descent approach
% % [Xest, obj, obj_ph, err, t] = phasemax(B,A,R,b,numiter,M,X0,beta,gamma,'GD', Xorig);
% 
% % using FASTA
% %  [Xest, obj_ph] = fasta_phasemax(C,R,b,numiter,M,L(1),X0,beta,gamma, Xorig);
% %  [Xest, obj_ph] = fasta_phasemax_vec(C,R,b,L,numiter,M,X0,beta,gamma, Xorig,lteq, eq);
% 
% % displaying the phase error
% if(strcmp(prob,'PM'))
%     disp(cellfun(@(Z) (angle(Z*Xest) - angle(Z*Xorig))*180/pi,R));
% end
% 
% %% GDRM algorithm
% 
% beta = 1;
% % numiter = 100;
% % [Xest, err3, t3] = gdrm(B,A,b,numiter,M,X0,diag(D),beta, Xorig);
% numiter = 1000;
% [X, obj_ph,t_gdrm,estimateMatError,absObjectiveError,itr] = gdrm_fasta(B,A,b,numiter,M,X0, beta,Xorig);
% gdrmOutput.matError = initialMatError;
% gdrmOutput.t =t_gdrm;
% gdrmOutput.iterations = itr;
% gdrmOutput.Xest=X;
% gdrmOutput.estimateMatError = estimateMatError;
% gdrmOutput.absObjectiveError = absObjectiveError;
% 
% 
% %% cvx solver
% [X] = solve_cvx(B,A,b,M,N,Xorig);
% 
% % end