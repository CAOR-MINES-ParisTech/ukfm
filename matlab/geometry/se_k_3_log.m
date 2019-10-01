function [xi] = se_k_3_log(chi)
%SE_K_3_LOG logarithm
%
% Syntax:  [xi] = se_k_3_log(chi)
%
% Inputs:
%    chi - matrix
%
% Outputs:
%    phi - vector

phi = so3_log(chi(1:3, 1:3));
Xi = so3_inv_left_jacobian(phi)*chi(1:3, 4:end);
xi = [phi;
     Xi(:)];
end