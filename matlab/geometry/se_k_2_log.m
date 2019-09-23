function [xi] = se_k_2_log(chi)
%SE_K_2_LOG logarithm map for SE_k(2)
%
% Syntax:  [xi] = se_k_2_log(chi)
%
% Inputs:
%    chi - state matrix
%
% Outputs:
%    xi - vector of length 1 + 2*k

phi = so2_log(chi(1:2, 1:2));
Xi = so2_inv_left_jacobian(phi)*chi(1:2, 3:end);
xi = [phi;
     Xi(:)];
end