function [chi_inv] = se_k_3_inv(chi)
%SE3_INV Inverse
%
% Syntax:  [chi_inv] = se_k_3_inv(chi)
%
% Inputs:
%    chi - matrix
%
% Outputs:
%    chi_inv - matrix

k = length(chi) - 3;
chi_inv = [chi(1:3, 1:3)' -chi(1:3, 1:3)'*chi(1:3, 4:end);
    zeros(k, 3) eye(k)];
end