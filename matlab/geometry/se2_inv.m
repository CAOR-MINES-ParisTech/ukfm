function [chi_inv] = se2_inv(chi)
%SE2_INV Inverse
%
% Syntax:  [chi_inv] = se2_inv(chi)
%
% Inputs:
%    chi - matrix
%
% Outputs:
%    chi_inv - matrix

chi_inv = [chi(1:2, 1:2)' -chi(1:2, 1:2)'*chi(1:2, 3);
    0 0 1];
end