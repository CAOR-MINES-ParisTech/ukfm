function [xi] = pendulum_phi_inv(state, hat_state)
%LOCALIZATION_PHI_INV inverse retraction
%
% Syntax: [xi] = pendulum_phi_inv(state, hat_state)
%
% Inputs:
%    state - state
%    hat_state - state
%
% Outputs:
%    xi - uncertainty

xi = [so3_log(hat_state.Rot' * state.Rot);
    state.u - hat_state.u];
end