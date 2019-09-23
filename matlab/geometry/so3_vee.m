function [phi] = so3_vee(Phi)
%SO3_VEE vee operator
%
% Syntax:  [phi] = so3_vee(Phi)
%
% Inputs:
%    phi - vector of length 3
%
% Outputs:
%    Phi - matrix of size 3x3

phi = [Phi(3, 2);
       Phi(1, 3);
       Phi(2, 1)];
end

