function [new_state] = inertial_navigation_right_phi(state, xi)
%INERTIAL_NAVIGATION_RIGHT_PHI retraction
%
% Syntax: [new_state] = inertial_navigation_right_phi(state, xi)
%
% Inputs:
%    state - state
%    xi - uncertainty
%
% Outputs:
%    new_state - state

chi = se_k_3_exp(xi);
new_state.Rot = chi(1:3, 1:3) * state.Rot;
new_state.v = chi(1:3, 1:3) * state.v + chi(1:3, 4);
new_state.p = chi(1:3, 1:3) * state.p + chi(1:3, 5);
end