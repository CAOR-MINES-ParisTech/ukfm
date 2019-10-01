function [errors] = inertial_navigation_errors(Rots, hat_Rots, vs, hat_vs, ...
    ps, hat_ps)
%INERTIAL_NAVIGATION_ERRORS
%
% Syntax: [errors] = inertial_navigation_errors(Rots, hat_Rots, vs, ...
%       hat_vs, ps, hat_ps)
%
% Inputs:
%    Rots - orientation matrices
%    hat_Rots - estimated orientation matrices
%    vs - velocities
%    hat_vs - estimated velocities
%    ps - positions
%    hat_ps - estimated positions
%
% Outputs:
%    errors - errors

N = length(Rots);
errors = zeros(9, N);
for n = 1:N
    errors(1:3, n) = so3_log(Rots{n}'*hat_Rots{n});
end
errors(4:6, :) = (vs-hat_vs)';
errors(7:9, :) = (ps-hat_ps)';
end

