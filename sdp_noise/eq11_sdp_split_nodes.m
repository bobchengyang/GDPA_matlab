function [cvx_optval] = eq11_sdp_split_nodes(label,b_ind,n_sample,dA,dB,dL)

db = 2*label(b_ind); % training labels x 2
dM = zeros(n_sample+1,n_sample+1); % number of total samples + 1
cvx_begin sdp
variables dy(n_sample+1,1) dz(length(b_ind),1);
minimize(sum(dy) + db'*dz)
subject to
for i=1:n_sample+1
    dM = dM + dy(i)*dA(:,:,i);
end
for i=1:length(b_ind)
    dM = dM + dz(i)*dB(:,:,i);
end
dM-dL>=0
% dM>=0
cvx_end

ml=full(dM-dL);

dy_old=dy;
dz_old=dz;
u=ml(n_sample+1,n_sample+1)+sum(dz_old); %u_{N+1}=y_{N+1}+sum(dz)

db = 2*label(b_ind); % training labels x 2
dz_plus_idx = db < 0;
dz_minus_idx = db > 0;

AL = zeros(n_sample+1+1);
Bnp1 = zeros(n_sample+1+1);
Bnp2 = zeros(n_sample+1+1);

b_ind_plus = b_ind(dz_plus_idx);
b_ind_minus = b_ind(dz_minus_idx);

cvx_begin sdp
variables dy(n_sample+1,1) dz(length(db),1);
minimize(sum(dy)+db'*dz)
subject to
for i=1:n_sample+2
    if i~=n_sample+2
      if i==n_sample+1
%     AL = AL + (0.5*u-sum(dz(b_ind_minus)))...
%         *[dA(:,:,i) zeros(n_sample+1,1); zeros(1,n_sample+1) 0];
    AL = AL + (0.5*dy(n_sample+1)+0.5*sum(dz(b_ind_plus))-0.5*sum(dz(b_ind_minus)))...
        *[dA(:,:,i) zeros(n_sample+1,1); zeros(1,n_sample+1) 0];
%     AL = AL + (dy(i))...
%         *[dA(:,:,i) zeros(n_sample+1,1); zeros(1,n_sample+1) 0];
      else
%     cL=-dL(1:n_sample,1:n_sample);    
%     off_dia_idx=1:n_sample;
%     off_dia_idx(i)=[];
%     AL = AL + (ml(i,i)+dz_old(i) - sum(cL(i,off_dia_idx)+dz(i)) )*[dA(:,:,i) zeros(n_sample+1,1); zeros(1,n_sample+1) 0];          
      
     AL = AL + (dy(i))*[dA(:,:,i) zeros(n_sample+1,1); zeros(1,n_sample+1) 0];          
      end
    else
    dAnp2 = zeros(n_sample+2);
    dAnp2(n_sample+2,n_sample+2)=1;
%     
%     AL = AL + (0.5*u-sum(dz(b_ind_plus)))...
%         *dAnp2;    
    AL = AL + (0.5*dy(n_sample+1)-0.5*sum(dz(b_ind_plus))+0.5*sum(dz(b_ind_minus)))...
        *dAnp2; 
%     AL = AL + (dy(i))...
%         *dAnp2; 
    end
end
for i=1:length(b_ind_minus)
    Bnp1 = Bnp1 + dz(b_ind_minus(i))*[dB(:,:,b_ind_minus(i)) zeros(n_sample+1,1);zeros(n_sample+1,1)' 0];
    dz(b_ind_minus(i))<0;
end
for i=1:length(b_ind_plus)
    dBnp2_i=dB(:,:,b_ind_plus(i));
    dBnp2_i=dBnp2_i(:,end);
    Bnp2 = Bnp2 + dz(b_ind_plus(i))*[zeros(n_sample+1) dBnp2_i;dBnp2_i' 0];
    dz(b_ind_plus(i))>0;
end
% dBnp2_i=zeros(n_sample+1,1);
% dBnp2_i(n_sample+1)=1;
% Bnp2=Bnp2+dz(n_sample+1)*[zeros(n_sample+1) dBnp2_i;dBnp2_i' 0];
AL+Bnp1+Bnp2-[dL zeros(n_sample+1,1); zeros(1,n_sample+1) 0]>=0;
cvx_end

H=AL+Bnp1+Bnp2-[dL zeros(n_sample+1,1); zeros(1,n_sample+1) 0];
H=(H+H')/2;

% % % fv_H_0=randn(n_sample+2,1);
% % % 
% % % [fv_H,lambda] = ...
% % %     lobpcg_fv(...
% % %     fv_H_0,...
% % %     H,...
% % %     1e-4,...
% % %     200);
% % % 
% % % x_val = sign(label(b_ind(1))*sign(fv_H(1)))*sign(fv_H(1:n_sample));
% % % % x_val = sign(fv_H(1:n_sample));
% % % x_pred=sign(x_val(round(n_sample/2)+1:1:n_sample));
% % % err_count = sum(abs(sign(x_val) - label))/2;

end

