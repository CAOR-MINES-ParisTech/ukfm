function [errors] = localization_errors(Rots, hat_Rots, ps, hat_ps)
%LOCALIZATION_ERRORS
%
% Syntax: [errors] = localization_errors(Rots, hat_Rots, ps, hat_ps)
%
% Inputs:
%    Rots - orientation matrices
%    hat_Rots - estimated orientation matrices
%    ps - positions
%    hat_ps - estimated positions
%
% Outputs:
%    errors - errors

N = length(Rots);
errors = zeros(3, N);
for n = 1:N
    errors(1, n) = so2_log(Rots{n}'*hat_Rots{n});
end
errors(2:3, :) = (ps-hat_ps)';
end

