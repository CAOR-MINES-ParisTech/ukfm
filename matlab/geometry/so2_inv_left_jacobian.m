function [J] = so2_inv_left_jacobian(phi)
%SO2_INV_LEFT_JACOBIAN inverse of Jacobian
%
% Syntax:  [J] = so2_inv_left_jacobian(phi)
%
% Inputs:
%    phi - scalar
%
% Outputs:
%    J - Jacobian

TOL = 1e-9;

if norm(phi) < TOL
    J = eye(2) - 1/2 * so2_wedge(phi);
else
    half_theta = phi/2;
    J = half_theta * cot(half_theta) * eye(2) - half_theta * so2_wedge(1);
end
end
