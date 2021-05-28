% *************************************************************
% author:   Gene Cheung                                      **
% date:     04/28/2021                                       **
% modified: 05/01/2021                                       **
% purpose:  test graph balance of SDP dual                   **
% *************************************************************

%clear all

%%% step 1: define A's, B's, graph Laplacian, and b *********
N = 5;
M = 3;
for i=1:N+1,
    v = [zeros(1,i-1) 1 zeros(1,N+1-i)];
    A(:,:,i) = diag(v);
    B(:,:,i) = zeros(N+1,N+1); 
    B(i,N+1,i) = 1;
    B(N+1,i,i) = 1;
    
    A2(:,:,i) = zeros(N+2,N+2);
    A2(i,i,i) = 1;
    B1(:,:,i) = zeros(N+2,N+2);
    B1(i,N+1,i) = 1;
    B1(N+1,i,i) = 1;
    B2(:,:,i) = zeros(N+2,N+2);
    B2(i,N+2,i) = 1;
    B2(N+2,i,i) = 1;
end
cL = [3 -1.5 0 0 -1; -1.5 4 -1 0 0; 0 -1 2 -1 0; 0 0 -1 2 -1; -1 0 0 -1 2];
L = [cL zeros(N,1); zeros(1,N) 0];
L2 = [cL zeros(N,2); zeros(2,N) zeros(2,2)]; 
b = 2*ones(M,1);
b(2) = -1*b(2);


if 1 == 0,
    %%% step 2b: run standard SDP dual to compute y, z********
    cvx_begin sdp
    variables y(N+1,1) z(M,1);
    minimize(ones(1,N+1)*y + b'*z)
    subject to
    % H = diag(y)+L;
    H = L;
    for i=1:N,
        H = H + y(i)*A(:,:,i);
    end
    H = H + y(N+1)*A(:,:,N+1);
    for i=1:M,
        H = H+z(i)*B(:,:,i);
    end
    H>=0;
    cvx_end 
    
else
    %%% step 2c: run new SDP dual to compute y, z********
    alpha = 0.5;        % parameter to split beweeen node2 N+1, N+2
    epsilon = 0.1;      % self-loop weight adjustment
    gamma = 10000;      % positive edge weight
    v = zeros(N+1,1);   % 1st eigenvector of H, to be computed
    converge = 0;
    while converge == 0
        cvx_begin sdp
        variables y(N+1,1) z(M,1);
        minimize(ones(1,N+1)*y + b'*z)
        subject to
        H = L2;
        for i=1:N,
            H = H + y(i)*A2(:,:,i);
        end
        H(N+1,N+1) = alpha*y(N+1) - epsilon;
        H(N+2,N+2) = (1-alpha)*y(N+1) + epsilon;
        for i=1:M,
            if b(i) > 0,  % thus z(i) < 0, i.e., positive edge
                H(N+1,N+1) = H(N+1,N+1) - alpha*z(i);
                H(N+2,N+2) = H(N+2,N+2) + (1-alpha)*z(i);
                H = H+z(i)*B1(:,:,i);
            else          % thus z(i) > 0, i.e., negative edge
                H(N+1,N+1) = H(N+1,N+1) + alpha*z(i);
                H(N+2,N+2) = H(N+2,N+2) - (1-alpha)*z(i);
                H = H+z(i)*B2(:,:,i);
            end
        end
        H = H - gamma*B2(:,:,N+1);
        H>=0;
        cvx_end  
    
        %%% reconstruct origH and H from SDP sol'n ********************
        origH = L;
        H = L2;        % reconstruct H
        for i=1:N,
            origH = origH + y(i)*A(:,:,i);
            H = H + y(i)*A2(:,:,i);
        end
        origH(N+1,N+1) = y(N+1);
        H(N+1,N+1) = alpha*y(N+1)-epsilon;
        H(N+2,N+2) = (1-alpha)*y(N+1)+epsilon;
        for i=1:M,
            origH = origH+z(i)*B(:,:,i);
            if b(i) > 0,  % thus z(i) < 0, i.e., positive edge
                H(N+1,N+1) = H(N+1,N+1) - alpha*z(i);
                H(N+2,N+2) = H(N+2,N+2) + (1-alpha)*z(i);
                H = H+z(i)*B1(:,:,i);
            else          % thus z(i) > 0, i.e., negative edge
                H(N+1,N+1) = H(N+1,N+1) + alpha*z(i);
                H(N+2,N+2) = H(N+2,N+2) - (1-alpha)*z(i);
                H = H+z(i)*B2(:,:,i);
            end
        end
        H = H - gamma*B2(:,:,N+1);
    
        %%% compute self-loop weight adjustments ********************
        [V,Lamb] = eig(origH);
        origLamb = Lamb(1,1);
        oldv = v;
        v = V(:,1);
        [V,Lamb] = eig(H);
        v2 = V(:,1);
        modLamb = Lamb(1,1);
        egap = origLamb - modLamb;
        [s,leftEnds] = compute_leftEnds(H);
        epsilon = epsilon*10;
        gamma = gamma*0.1;
%         S1 = 0;
%         S2 = 0;
%         for i=1:M,
%             if z(i) < 0,    % positive edges
%                 S1 = S1 - z(i)*(v(N+1) - v(i));
%             else            % negative edges
%                 S2 = S2 - z(i)*(v(N+1) - v(i));
%             end
%         end
%         epsilon = (S1 - S2)/(2*v(N+1));     % self-loop weight adjustment
        converge = (norm(oldv -v)<0.0001);
    end
    
end


%%% step 3: review PSD matrix H ********
obj = ones(1,N+1)*y + b'*z;
%obj = ones(1,N)*y(1:N) + beta^2*y(N+1) + beta*b'*z;
[V,Lamb] = eig(H);
v = V(:,1);
