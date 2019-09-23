function [chi_inv] = se_k_3_inv(chi)
%SE3_INV Inverse for SE_k(3)
%
% Syntax:  [chi_inv] = se_k_3_inv(chi)
%
% Inputs:
%    chi - state matrix
%
% Outputs:
%    chi_inv - state matrix

k = length(chi) - 3;
chi_inv = [chi(1:3, 1:3)' -chi(1:3, 1:3)'*chi(1:3, 4:end);
    zeros(k, 3) eye(k)];
end