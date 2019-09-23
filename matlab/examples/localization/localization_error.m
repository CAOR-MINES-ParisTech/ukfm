function [errors] = localization_error(states, hat_states)
%LOCALIZATION_ERROR Compute metrics
%
% Syntax: [errors] = localization_error(states, hat_states)
%
% Inputs:
%    states - true states
%    hat_states - estimated states
%
% Outputs:
%    errors - orientation and position mean square errors

% get true state
state_table = struct2table(states);
Rot = state_table.Rot;
N = length(Rot);

% get estimated state
kf_state_table = struct2table(hat_states);
kf_Rot = kf_state_table.Rot;
error_rot = zeros(1);
for n = 1:N
    error_rot = error_rot + so2_log(Rot{n}'*kf_Rot{n})^2;
end
errors = error_rot;
end

