function [X_0] = initialize_biconvex_X(C, L, b, m_epsilon_start, m_epsilon_end, m_cons_same, m)

n = size(C, 1);
X_0 = randn(n, m);


