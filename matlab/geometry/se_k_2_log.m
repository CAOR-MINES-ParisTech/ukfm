function [xi] = se_k_2_log(chi)
%SE_K_2_LOG logarithm
%
% Syntax:  [xi] = se_k_2_log(chi)
%
% Inputs:
%    chi - matrix
%
% Outputs:
%    xi - vector

phi = so2_log(chi(1:2, 1:2));
Xi = so2_inv_left_jacobian(phi)*chi(1:2, 3:end);
xi = [phi;
     Xi(:)];
end