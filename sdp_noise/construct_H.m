function [initial_H] = construct_H(sw,n_sample,...
    cL,...
    u,...
    alpha,...
    dy_LP_test_init,...
    dz_LP_test_init,...
    dz_ind_plus,...
    dz_ind_minus,...
    mode,...
    gamma)
initial_H=zeros(n_sample+2);
if mode==1
    initial_H(1:n_sample,1:n_sample)=diag(dy_LP_test_init)+cL;
    initial_H(n_sample+1,n_sample+1)=-sum(dz_LP_test_init(dz_ind_minus))+0.5*u(n_sample+1)-alpha;
    initial_H(n_sample+2,n_sample+2)=-sum(dz_LP_test_init(dz_ind_plus))+0.5*u(n_sample+1)+alpha;
elseif mode==2
    initial_H(1:n_sample,1:n_sample)=cL;
    initial_H(1:n_sample+2+1:end)=initial_H(1:n_sample+2+1:end)+dy_LP_test_init';
elseif mode==3
    initial_H(1:n_sample,1:n_sample)=cL;
    initial_H(1:n_sample+2+1:end)=initial_H(1:n_sample+2+1:end)+...
        [dy_LP_test_init(1:n_sample)' ...
        sw*(dy_LP_test_init(n_sample+1)+sum(dz_LP_test_init))-sum(dz_LP_test_init(dz_ind_minus))-alpha ...
        (1-sw)*(dy_LP_test_init(n_sample+1)+sum(dz_LP_test_init))-sum(dz_LP_test_init(dz_ind_plus))+alpha];
elseif mode==4
    initial_H(1:n_sample,1:n_sample)=cL;
    initial_H(1:n_sample+2+1:end)=initial_H(1:n_sample+2+1:end)+...
        [dy_LP_test_init(1:n_sample)' ...
        sw*(dy_LP_test_init(n_sample+1)+sum(dz_LP_test_init))-sum(dz_LP_test_init(dz_ind_minus))-alpha ...
        (1-sw)*(dy_LP_test_init(n_sample+1)+sum(dz_LP_test_init))-sum(dz_LP_test_init(dz_ind_plus))+alpha]; 
    initial_H(n_sample+1,n_sample+2)=-gamma;
    initial_H(n_sample+2,n_sample+1)=-gamma;
end
initial_H(dz_ind_minus,n_sample+1)=dz_LP_test_init(dz_ind_minus);
initial_H(n_sample+1,dz_ind_minus)=dz_LP_test_init(dz_ind_minus);
initial_H(dz_ind_plus,n_sample+2)=dz_LP_test_init(dz_ind_plus);
initial_H(n_sample+2,dz_ind_plus)=dz_LP_test_init(dz_ind_plus);

%% adding a small positive edge weight between nodes N+1 and N+2
% initial_H(n_sample+2,n_sample+1)=-1e-15;
% initial_H(n_sample+1,n_sample+2)=-1e-15;
end
