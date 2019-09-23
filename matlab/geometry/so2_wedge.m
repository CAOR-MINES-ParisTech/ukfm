function [Phi] = so2_wedge(phi)
%SO2_WEDGE Wedge operator
%
% Syntax:  [Phi] = so2_wedge(phi)
%
% Inputs:
%    phi - scalar

%
% Outputs:
%    Phi - matrix

Phi = [0 -phi;
      phi 0];
end

