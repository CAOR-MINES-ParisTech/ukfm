function [xi] = se2_log(chi)
%SE2_LOG logarithm map for SE(2)
%
% Syntax:  [xi] = se2_log(chi)
%
% Inputs:
%    chi - state matrix
%
% Outputs:
%    xi - vector of length 3

phi = so2_log(chi(1:2, 1:2));
xi = [phi;
     so2_inv_left_jacobian(phi)*chi(1:2, 3)];
end
