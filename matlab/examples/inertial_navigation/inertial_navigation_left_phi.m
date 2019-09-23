function [new_state] = inertial_navigation_left_phi(state, xi)
%INERTIAL_NAVIGATION_LEFT_PHI retraction
%
% Syntax: [new_state] = inertial_navigation_left_phi(state, xi)
%
% Inputs:
%    state - state
%    xi - uncertainty
%
% Outputs:
%    new_state - state

chi = se_k_3_exp(xi);
new_state.Rot = state.Rot * chi(1:3, 1:3);
new_state.v = state.Rot * chi(1:3, 4) + state.v;
new_state.p = state.Rot * chi(1:3, 5) + state.p;
end