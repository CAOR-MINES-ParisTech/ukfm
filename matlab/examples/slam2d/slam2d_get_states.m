function [Rots, ps] = slam2d_get_states(states)
%SLAM2D_GET_STATES
%
% Syntax: [Rots, ps] = slam2d_get_states(states)
%
% Inputs:
%    states - states
% Outputs:
%    Rots - orientation matrices
%    ps - positions

state_table = struct2table(states);
Rots = state_table.Rot;
ps = cell2mat(state_table.p')';
end

