function [xi] = localization_phi_inv(state, hat_state)
%LOCALIZATION_PHI_INV inverse retraction
%
% Syntax: [xi] = localization_phi_inv(state, hat_state)
%
% Inputs:
%    state - state
%    hat_state - state
%
% Outputs:
%    xi - uncertainty

xi = [so2_log(hat_state.Rot * state.Rot');
    hat_state.p - state.p];
end