function [phi] = so3_vee(Phi)
%SO3_VEE vee operator
%
% Syntax:  [phi] = so3_vee(Phi)
%
% Inputs:
%    phi - vector
%
% Outputs:
%    Phi - matrix

phi = [Phi(3, 2);
       Phi(1, 3);
       Phi(2, 1)];
end

