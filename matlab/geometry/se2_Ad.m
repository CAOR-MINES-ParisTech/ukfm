function [Ad] = se2_Ad(chi)
%SE2_EXP Adjoint matrix of SE(2)
%
% Syntax: [Ad] = se2_Ad(chi)
%
% Inputs:
%    chi - state matrix
%
% Outputs:
%    Ad - adjoint matrix

Rot = chi(1:2, 1:2);
Jp = [chi(2, 3);
      -chi(1, 3)];
Ad = [Jp, Rot;
      1 0 0];
end

