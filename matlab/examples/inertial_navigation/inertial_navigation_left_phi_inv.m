function [xi] = inertial_navigation_left_phi_inv(state, hat_state)
%INERTIAL_NAVIGATION_LEFT_PHI_INV inverse retraction
%
% Syntax: [xi] = inertial_navigation_left_phi_inv(state, hat_state)
%
% Inputs:
%    state - state
%    hat_state - state
%
% Outputs:
%    xi - uncertainty

chi = [state.Rot state.v state.p;
    zeros(2, 3) eye(2)];
hat_chi = [hat_state.Rot hat_state.v hat_state.p;
    zeros(2, 3) eye(2)];
xi = se_k_3_log(se_k_3_inv(chi) * hat_chi);
end