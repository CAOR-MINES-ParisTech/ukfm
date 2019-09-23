function [chi_inv] = se2_inv(chi)
%SE2_INV Inverse for SE(2)
%
% Syntax:  [chi_inv] = se2_inv(chi)
%
% Inputs:
%    chi - state matrix
%
% Outputs:
%    chi_inv - state matrix

chi_inv = [chi(1:2, 1:2)' -chi(1:2, 1:2)'*chi(1:2, 3);
    0 0 1];
end