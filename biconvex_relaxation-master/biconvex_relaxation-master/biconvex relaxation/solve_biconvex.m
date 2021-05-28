function [cut, x_opt, x_real] = solve_sdcut(A, options)

    disp('-----------------------------------------------');
    disp('biconvex_solve_fast start ...');

    % scale A
    norm_A = sqrt(sum(sum(A.^2)));
    A = A ./ norm_A;
    
    
    % get B, b, m
    [B, B_non_sparse_part, b, c, a, gs, kappas, l_bbox, u_bbox, m, B_non_sparse_part_fast] ...
        = sdcut_pack_cons_data(A, options);
    [X, obj, obj_ph, t] = phasemax_imseg(A,B,b,1000,2,2,'G',m,5,false);
    

    % recover x from X = x * x'
    x_opt = X; 
    x_real=0;

    n = size(A, 1);
    cut = zeros(n, 2);
    cut(:,1) = x_opt == 1;
    cut(:,2) = x_opt == -1;

    disp('biconvext_solve_fast end');
    disp('-----------------------------------------------');
end

function u_init = calc_u_init(A, options, m)

    u_init = zeros(m,1);
    
end

function [X, obj, obj_ph, t] = phasemax_imseg(C,R,b,numiter,beta,gamma,method,N,alpha,flag)
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

function [u_opt, iters] = solve_dual_lbfgsb_v3(u_init, A, B, b, l_bbox, u_bbox, ...
    sigma, B_non_sparse_part, lbfgsb_factr, lbfgsb_pgtol, lbfgsb_m, B_non_sparse_part_fast)

    fcn = @(u) calc_dual_obj_grad_lbfgsb(u, A, B, b, sigma, B_non_sparse_part, B_non_sparse_part_fast);

    opts = struct('x0', u_init, 'maxIts', 1000, ...
        'factr', lbfgsb_factr, 'pgtol', lbfgsb_pgtol, 'm', lbfgsb_m);

    m = length(u_init);

    [u_opt, ~, info] = lbfgsb(fcn, l_bbox, u_bbox, opts);

    iters = info.totalIterations;

end


function [obj, grad] = calc_dual_obj_grad_lbfgsb(u, A, B, b, sigma, B_non_sparse_part, B_non_sparse_part_fast)

    %% calc C_minus
    persistent pre_V;
    persistent pre_D;

    num_neg_eigen = length(find(diag(pre_D) < 0));
    % 
    if isempty(pre_D) || num_neg_eigen > 100
        [C_minus, pre_V, pre_D] = calc_c_minus(u, A, B);
    else
        C = [];    

        eigen_opts.eigen_solver = 'eigs';
        eigen_opts.pre_V = pre_V;
        eigen_opts.pre_D = pre_D;
        eigen_opts.num_neg_eigen = num_neg_eigen + 2;
        eigen_opts.A_fun = @(x) calc_Cx(u, A, B, x, B_non_sparse_part, B_non_sparse_part_fast);

        [~, C_minus, pre_V, pre_D] = calc_pos_neg_part(C, eigen_opts); 
    end

    %% calc objective
    obj = calc_dual_obj(u, A, B, b, sigma, C_minus);

    %% calc gradient
    m = length(u);
    grad = zeros(m, 1);
    for ii = 1 : m
        grad(ii) = C_minus(:)' * B{ii}(:);
    end
    grad = (0.5 / sigma) * grad + b;

end


function Cx = calc_Cx(u, A, B, x, B_non_sparse_part, B_non_sparse_part_fast)

    n = size(A,1);


    % sparse part
    Cx = (sparse(1:n, 1:n, u(1:n)*B{1}(1,1), n, n) - A) * x;
    
    
    % structural part
    V  = B_non_sparse_part_fast.V;
    dd = B_non_sparse_part_fast.dd;
    Cx = Cx + V * (u(n+1:end) .* dd .* (V'*x));

end


function [C_minus, V, D] = calc_c_minus(u, A, B)

    m = length(B);

    C = -1 * A;
    for ii = 1 : m
        C = C + u(ii) * B{ii};
    end

    [~, C_minus, V, D] = calc_pos_neg_part(C, struct('eigen_solver', 'eig'));

end

function obj = calc_primal_obj(X, A, sigma)

    obj = -1 * A(:)' * X(:) + sigma * X(:)' * X(:);

end

function obj = calc_dual_obj(u, A, B, b, sigma, C_minus)

    if nargin == 5
        C_minus = calc_c_minus(u, A, B);
    end
        
    obj = (-0.25 / sigma) * sum(C_minus(:) .^ 2)  - u' * b;
    obj = -1 * obj;

end


function [X_plus, X_minus, V, D] = calc_pos_neg_part(X, options)


if nargin == 1
    options.eigen_solver = 'eig';
end


% make sure X is symmetric
% X = (X + X') / 2;

% eigenvectors and eigenvalues
switch options.eigen_solver
    case 'eig'
        [V, D] = eig(X);
    case 'eigs'
        max_d = -inf;
        k = options.num_neg_eigen;
        [n, K] = size(options.pre_V);
        eigs_opts.issym  = 1;
        eigs_opts.isreal = 1;  
        
        eigs_opts.v0 = options.pre_V * ones(K, 1);
        % eigs_opts.v0 = options.pre_V * rand(K, 1);
        % eigs_opts.v0 = options.pre_V(:,1);
            
        while max_d < 0
            eigs_opts.p = max(k, min(n, 2 * k + 10));
            fprintf(1, 'eigs start, k = %d ...\n', k);
            [V, D] = eigs(options.A_fun, n, k, 'sa', eigs_opts);
%             [V, D] = eigs(X, k, 'sa', eigs_opts);
            fprintf(1, 'eigs end\n');            
            
            max_d = max(diag(D));
            if max_d < 0
                min_d = min(diag(D));
                fprintf('max_d == %f < 0, min_d = %f, k = %d, n = %d\n', max_d, min_d, k, n);
                k = min(n, k*2);
            end
        end

    otherwise
        error('unknown eigen_solver: %s\n', eigen_solver);
end

dd = real( diag(D) );
idxs_minus = find(dd < 0);
n_minus = length(idxs_minus);
D_minus = sparse(1:n_minus, 1:n_minus, dd(idxs_minus), n_minus, n_minus, n_minus); 
V_minus = V(:, idxs_minus);
X_minus = (V_minus * D_minus) * V_minus';

fprintf('rank of negative part: %d, pos part: %d\n', n_minus, length(find(dd>0)));

if ~isempty(X)
    X_plus = X - X_minus;

    X_diff = max(max(abs(X - (X_plus + X_minus)))) / max(abs(X(:)));
    if max(X_diff(:)) > 0.001
        fprintf(1, 'X ~= X_plus + X_minus, diff: %f\n', max(X_diff(:)));
    end
else
    X_plus = [];
end

end

