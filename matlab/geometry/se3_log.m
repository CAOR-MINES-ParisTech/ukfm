function [xi] = se3_log(chi)
%SE3_LOG logarithm
%
% Syntax:  [xi] = se3_log(chi)
%
% Inputs:
%    chi - matrix
%
% Outputs:
%    xi - vector

phi = so3_log(chi(1:3, 1:3));
xi = [phi;
     so3_inv_left_jacobian(phi)*chi(1:3, 4)];
end