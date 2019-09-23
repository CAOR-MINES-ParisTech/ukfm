function [Phi] = so3_wedge(phi)
%SO3_WEDGE Wedge operator
%
% Syntax:  [Phi] = so3_wedge(phi)
%
% Inputs:
%    phi - vector of length 3
%
% Outputs:
%    Phi - matrix of size 3x3

Phi = [0, -phi(3), phi(2);
       phi(3), 0, -phi(1);
       -phi(2), phi(1), 0];
end
