function [errors] = attitude_errors(Rots, hat_Rots)
%ATTITUDE_ERRORS
%
% Syntax: [errors] = attitude_errors(Rots, hat_Rots)
%
% Inputs:
%    Rots - orientation matrices
%    hat_Rots - estimated orientation matrices
%
% Outputs:
%    errors - errors

N = length(Rots);
errors = zeros(3, N);
for n = 1:N
    errors(:, n) = so3_log(Rots{n}'*hat_Rots{n});
end
end

