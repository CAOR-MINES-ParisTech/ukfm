function [chi_inv] = se3_inv(chi)
%SE3_INV Inverse for SE(3)
%
% Syntax:  [chi_inv] = se3_inv(chi)
%
% Inputs:
%    chi - state matrix
%
% Outputs:
%    chi_inv - state matrix

chi_inv = [chi(1:3, 1:3)' -chi(1:3, 1:3)'*chi(1:3, 4);
    0 0 0 1];
end