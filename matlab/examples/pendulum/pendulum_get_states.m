function [Rots, us] = pendulum_get_states(states)
%PENDULUM_GET_STATES
%
% Syntax: [Rots, us] = pendulum_get_states(states)
%
% Inputs:
%    states - states
% Outputs:
%    Rots - orientation matrices
%    us - angular velocities

state_table = struct2table(states);
Rots = state_table.Rot;
us = cell2mat(state_table.u')';
end
