function Rots = attitude_get_states(states)
%ATTITUDE_GET_STATES
%
% Syntax: Rots = attitude_get_states(states)
%
% Inputs:
%    states - states
% Outputs:
%    Rots - orientation matrices

state_table = struct2table(states);
Rots = state_table.Rot;
end

