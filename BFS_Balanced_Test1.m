A=[0 0.1 -1 1 -1 1 0 0; 0.1 0 1 0 1 0 0 0; -1 1 0 -1 0 0 0 0; 1 0 -1 0 0 0 1 -1; -1 1 0 0 0 1 0 0; 1 0 0 0 1 0 -1 0; 0 0 0 1 0 -1 0 -1; 0 0 0 -1 0 0 -1 0];

G=graph(A);

figure(1);
plot(G,'EdgeLabel', G.Edges.Weight);
title('Unbalanced Graph');

A_B=BFS_Balanced(A);

G2=graph(A_B);
figure(2);

plot(G2,'EdgeLabel',G2.Edges.Weight);
title('balanced Graph');
