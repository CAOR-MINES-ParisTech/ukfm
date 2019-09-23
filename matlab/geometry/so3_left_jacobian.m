function [J] = so3_left_jacobian(phi)
%SO3_LEFT_JACOBIAN Jacobian for SO(3)
%
% Syntax:  [J] = so3_left_jacobian(phi)
%
% Inputs:
%    phi - vector of length 3
%
% Outputs:
%    J - Jacobian, matrix of size 3x3

TOL = 1e-9;

angle = norm(phi);
if angle < TOL
    % Near |phi|==0, use first order Taylor expansion
    J = eye(3) - 1/2 * so3_wedge(phi);
else
    axis = phi / angle;
    s = sin(angle);
    c = cos(angle);
    J = (s / angle) * eye(3) + (1 - s / angle) * (axis*axis') + ...
            ((1 - c) / angle) * so3_wedge(axis);
end
end