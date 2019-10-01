function [Phi] = so3_wedge(phi)
%SO3_WEDGE Wedge operator
%
% Syntax:  [Phi] = so3_wedge(phi)
%
% Inputs:
%    phi - vector
%
% Outputs:
%    Phi - matrix

Phi = [0, -phi(3), phi(2);
       phi(3), 0, -phi(1);
       -phi(2), phi(1), 0];
end
