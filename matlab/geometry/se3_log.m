function [xi] = se3_log(chi)
%SE3_LOG logarithm map for SE(3)
%
% Syntax:  [xi] = se3_log(chi)
%
% Inputs:
%    chi - state matrix
%
% Outputs:
%    xi - vector of length 6

phi = so3_log(chi(1:3, 1:3));
xi = [phi;
     so3_inv_left_jacobian(phi)*chi(1:3, 4)];
end