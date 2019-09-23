function [Rot] = so3_exp(phi)
%SO3_EXP exponential map for SO(3)
%
% Syntax:  [Rot] = so3_exp(phi)
%
% Inputs:
%    phi - vector of length 3
%
% Outputs:
%    Rot - rotation matrix

TOL = 1e-9;

angle = norm(phi);
if angle < TOL
    % Near |phi|==0, use first order Taylor expansion
    Rot = eye(3) + so3_wedge(phi);
else
    axis = phi / angle;
    Rot = cos(angle) * eye(3) + (1-cos(angle))*(axis*axis') + ...
        sin(angle) * so3_wedge(axis);
end
