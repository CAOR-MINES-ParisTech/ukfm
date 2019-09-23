function [errors] = inertial_navigation_error(states, hat_states)
%INERTIAL_NAVIGATION_ERROR Compute metrics
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
p = cell2mat(state_table.p')';
N = length(p);

% get estimated state
kf_state_table = struct2table(hat_states);
kf_Rot = kf_state_table.Rot;
kf_p = cell2mat(kf_state_table.p')';
error_rot = zeros(1);
for n = 1:N
    error_rot = error_rot + norm(so3_log(Rot{n}'*kf_Rot{n}))^2;
end

errors = [error_rot; 
    norm(p-kf_p)^2];
end

