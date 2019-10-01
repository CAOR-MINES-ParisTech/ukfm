function [new_state] = pendulum_phi(state, xi)
%pendulum_PHI retraction
%
% Syntax: [new_state] = pendulum_phi(state, xi)
%
% Inputs:
%    state - state
%    xi - uncertainty
%
% Outputs:
%    new_state - state

new_state.Rot = state.Rot * so3_exp(xi(1:3));
new_state.u = state.u + xi(4:6);
end