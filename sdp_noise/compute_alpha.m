function [alpha] = compute_alpha(...
    n_sample,...
    db,...
    b_ind,...
    original_H,...
    fv_H)
db_plus_idx = db > 0;
db_minus_idx = db < 0;
db_plus_idx = b_ind(db_plus_idx);
db_minus_idx = b_ind(db_minus_idx);

w_positive=original_H(db_plus_idx,n_sample+1); % corres. to db_plus_idx
w_positive=-w_positive;
w_negative=original_H(db_minus_idx,n_sample+1); % corres. to db_minus_idx
w_negative=-w_negative;
delta_xNplus1_xi_positive=fv_H(n_sample+1)-fv_H(db_plus_idx);
delta_xNplus1_xi_negative=fv_H(n_sample+1)-fv_H(db_minus_idx);
sNplus1=sum(w_positive.*delta_xNplus1_xi_positive);
sNplus2=sum(w_negative.*delta_xNplus1_xi_negative);
alpha=(sNplus1-sNplus2)/(2*fv_H(end));
end

