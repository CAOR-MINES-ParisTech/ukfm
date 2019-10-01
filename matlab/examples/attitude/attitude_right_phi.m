function [new_state] = attitude_right_phi(state, xi)
%ATTITUDE_RIGHT_PHI retraction
%
% Syntax: [new_state] = attitude_right_phi(state, xi)
%
% Inputs:
%    state - state
%    xi - uncertainty
%
% Outputs:
%    new_state - state

new_state.Rot = so3_exp(xi) * state.Rot;
end