function [new_state] = localization_phi(state, xi)
%LOCALIZATION_PHI retraction
%
% Syntax: [new_state] = localization_phi(state, xi)
%
% Inputs:
%    state - state
%    xi - uncertainty
%
% Outputs:
%    new_state - state

new_state.Rot = state.Rot * so2_exp(xi(1));
new_state.p = state.p + xi(2:3);
end