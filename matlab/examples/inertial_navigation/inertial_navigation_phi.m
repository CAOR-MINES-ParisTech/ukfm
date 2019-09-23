function [new_state] = inertial_navigation_phi(state, xi)
%INERTIAL_NAVIGATION_PHI retraction
%
% Syntax: [new_state] = inertial_navigation_phi(state, xi)
%
% Inputs:
%    state - state
%    xi - uncertainty
%
% Outputs:
%    new_state - state

new_state.Rot = so3_exp(xi(1:3)) * state.Rot;
new_state.v = state.v + xi(4:6);
new_state.p = state.p + xi(7:9);
end

