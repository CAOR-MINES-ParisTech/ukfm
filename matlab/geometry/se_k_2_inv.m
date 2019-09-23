function [chi_inv] = se_k_2_inv(chi)
%SE3_INV Inverse for SE_k(2)
%
% Syntax:  [chi_inv] = se_k_2_inv(chi)
%
% Inputs:
%    chi - state matrix
%
% Outputs:
%    chi_inv - state matrix

k = length(chi) - 2;
chi_inv = [chi(1:2, 1:2)' -chi(1:2, 1:2)'*chi(1:2, 3:end);
    zeros(k, 2) eye(k)];
end