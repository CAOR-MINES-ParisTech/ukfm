function [xi] = se_k_3_log(chi)
%SE_K_3_LOG logarithm map for SE_k(3)
%
% Syntax:  [xi] = se_k_3_log(chi)
%
% Inputs:
%    chi - state matrix
%
% Outputs:
%    phi - vector of length 3*(k+1)

phi = so3_log(chi(1:3, 1:3));
Xi = so3_inv_left_jacobian(phi)*chi(1:3, 4:end);
xi = [phi;
     Xi(:)];
end