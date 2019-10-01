function [chi_inv] = se3_inv(chi)
%SE3_INV Inverse
%
% Syntax:  [chi_inv] = se3_inv(chi)
%
% Inputs:
%    chi - matrix
%
% Outputs:
%    chi_inv - matrix

chi_inv = [chi(1:3, 1:3)' -chi(1:3, 1:3)'*chi(1:3, 4);
    0 0 0 1];
end