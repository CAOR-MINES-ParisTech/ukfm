function [errors] = slam2d_error(state, kf_state)
%SLAM2D_ERROR Compute metrics
%
% Syntax: [errors] = slam2d_error(state, kf_state)
%
% Inputs:
%    state - true state
%    kf_state - estimated state
%
% Outputs:
%    errors - orientation and position mean square errors

% get true state
state_table = struct2table(state);
Rot = state_table.Rot;
p = cell2mat(state_table.p')';
N = length(p);

% get estimated state
kf_state_table = struct2table(kf_state);
kf_Rot = kf_state_table.Rot;
kf_p = cell2mat(kf_state_table.p')';
error_rot = zeros(1);
% for n = 1:N
%     error_rot = error_rot + so2_log(Rot{n}'*kf_Rot{n})^2;
% end
errors = [error_rot; 
    norm(p-kf_p)^2];

end

