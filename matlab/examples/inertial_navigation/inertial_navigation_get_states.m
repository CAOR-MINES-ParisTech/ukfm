function [Rots, vs, ps] = inertial_navigation_get_states(states)
%INERTIAL_NAVIGATION_GET_STATES
%
% Syntax: [Rots, vs, ps] = inertial_navigation_get_states(states)
%
% Inputs:
%    states - states
% Outputs:
%    Rots - orientation matrices
%    vs - velocities
%    ps - positions

state_table = struct2table(states);
Rots = state_table.Rot;
vs = cell2mat(state_table.v')';
ps = cell2mat(state_table.p')';
end

