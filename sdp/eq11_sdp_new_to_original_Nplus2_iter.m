function [cvx_optval,eigen_gap] = eq11_sdp_new_to_original_Nplus2_iter(label,b_ind,n_sample,dA,dB,dL,scale,u,alpha)

db = 2*label(b_ind); % training labels x 2
dz_plus_idx = db < 0;
dz_minus_idx = db > 0;

AL = zeros(n_sample+1+1);
Bnp1 = zeros(n_sample+1+1);
Bnp2 = zeros(n_sample+1+1);

b_ind_plus = b_ind(dz_plus_idx);
b_ind_minus = b_ind(dz_minus_idx);

cL=-dL(1:n_sample,1:n_sample);
cL_offdia=cL;
cL_offdia(1:n_sample+1:end)=0;

cvx_begin sdp
variables dy(n_sample+1,1) dz(length(b_ind),1);
minimize(sum(dy)+db'*dz)
subject to
% alpha=(sum(-dz(b_ind_minus).*2*scale)-sum(-dz(b_ind_plus).*2*scale))/(2*scale);
% for ii=1:n_sample
%     if ii<=length(b_ind)
%     dy(ii)+cL(ii,ii)==-sum(cL_offdia(ii,:))-dz(ii)+u(ii);
%     else
%     dy(ii)+cL(ii,ii)==-sum(cL_offdia(ii,:))+u(ii);    
%     end
% end
% dy(n_sample+1)==-sum(dz(b_ind_minus))+u(n_sample+1)-alpha;
% dy_node_Nplus2==-sum(dz(b_ind_plus))+u(n_sample+1)+alpha;
for i=1:n_sample+2
    if i~=n_sample+2
        if i==n_sample+1
%             AL = AL + dy(i)...
%                 *[dA(:,:,i) zeros(n_sample+1,1); zeros(1,n_sample+1) 0];
            AL = AL + (0.5*dy(n_sample+1)+0.5*sum(dz(b_ind_plus))-0.5*sum(dz(b_ind_minus)))...
                *[dA(:,:,i) zeros(n_sample+1,1); zeros(1,n_sample+1) 0];
%             AL = AL + (dy(i))...
%                 *[dA(:,:,i) zeros(n_sample+1,1); zeros(1,n_sample+1) 0];
        else
            AL = AL + (dy(i))*[dA(:,:,i) zeros(n_sample+1,1); zeros(1,n_sample+1) 0];
        end
    else
        dAnp2 = zeros(n_sample+2);
        dAnp2(n_sample+2,n_sample+2)=1;
%         AL = AL + dy(i)...
%             *dAnp2;        
        AL = AL + (0.5*dy(n_sample+1)-0.5*sum(dz(b_ind_plus))+0.5*sum(dz(b_ind_minus)))...
            *dAnp2;
%         AL = AL + (dy_node_Nplus2)...
%             *dAnp2;
    end
end
for i=1:length(b_ind_minus)
%     Bnp1=Bnp1+dz(i)*[dB(:,:,i) zeros(n_sample+1,1);zeros(n_sample+1,1)' 0];
    Bnp1 = Bnp1 + dz(b_ind_minus(i))*[dB(:,:,b_ind_minus(i)) zeros(n_sample+1,1);zeros(n_sample+1,1)' 0];
%     dz(i)<0;
    dz(b_ind_minus(i))<0;
end
% for i=1:length(b_ind_minus)
% %     Bnp1=Bnp1+dz(i)*[dB(:,:,i) zeros(n_sample+1,1);zeros(n_sample+1,1)' 0];
%     Bnp1 = Bnp1 + dz(n_sample+b_ind_minus(i))*[dB(:,:,b_ind_minus(i)-1) zeros(n_sample+1,1);zeros(n_sample+1,1)' 0];
% %     dz(i)<0;
%     dz(n_sample+b_ind_minus(i))<0;
% end

for i=1:length(b_ind_plus)
%     dBnp2_i=dB(:,:,i);
    dBnp2_i=dB(:,:,b_ind_plus(i));
    dBnp2_i=dBnp2_i(:,end);
%     Bnp2=Bnp2+dz(n_sample+i)*[zeros(n_sample+1) dBnp2_i;dBnp2_i' 0];
    Bnp2 = Bnp2 + dz(b_ind_plus(i))*[zeros(n_sample+1) dBnp2_i;dBnp2_i' 0];
%     dz(n_sample+i)>0;
    dz(b_ind_plus(i))>0;
end

% for i=1:length(b_ind_plus)
% %     dBnp2_i=dB(:,:,i);
%     dBnp2_i=dB(:,:,b_ind_plus(i)+1);
%     dBnp2_i=dBnp2_i(:,end);
% %     Bnp2=Bnp2+dz(n_sample+i)*[zeros(n_sample+1) dBnp2_i;dBnp2_i' 0];
%     Bnp2 = Bnp2 + dz(n_sample+b_ind_plus(i))*[zeros(n_sample+1) dBnp2_i;dBnp2_i' 0];
% %     dz(n_sample+i)>0;
%     dz(n_sample+b_ind_plus(i))>0;
% end

% Bnp3=zeros(n_sample+2);
% Bnp3(end-1,end)=1;
% Bnp3(end,end-1)=1;

% dd>=0;

AL+Bnp1+Bnp2-[dL zeros(n_sample+1,1); zeros(1,n_sample+1) 0]>=0;
cvx_end

new_H=AL+Bnp1+Bnp2-[dL zeros(n_sample+1,1); zeros(1,n_sample+1) 0];
new_H=(new_H+new_H')/2;

original_H=-dL;

original_H(1:n_sample+1+1:end)=diag(-dL)+dy;
original_H(1:n_sample,end)=dz;
original_H(end,1:n_sample)=dz;

original_H=(original_H+original_H')/2;
new_H=(new_H+new_H')/2;

eigen_gap=min(eig(new_H))-min(eig(original_H));
end