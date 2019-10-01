function [xi] = inertial_navigation_phi_inv(state, hat_state)
%INERTIAL_NAVIGATION_PHI_INV inverse retraction
%
% Syntax: [xi] = inertial_navigation_phi_inv(state, hat_state)
%
% Inputs:
%    state - state
%    hat_state - state
%
% Outputs:
%    xi - uncertainty

xi = [so3_log(state.Rot' * hat_state.Rot);
    hat_state.v - state.v;
    hat_state.p - state.p];
end

