function [new_state] = attitude_phi(state, xi)
%ATTITUDE_PHI retraction
%
% Syntax: [new_state] = attitude_phi(state, xi)
%
% Inputs:
%    state - state
%    xi - uncertainty
%
% Outputs:
%    new_state - state

new_state.Rot = state.Rot * so3_exp(xi);
end

