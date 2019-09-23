function [xi] = sphere_left_phi_inv(state, hat_state)
%LOCALIZATION_LEFT_PHI_INV inverse retraction
%
% Syntax: [xi] = sphere_left_phi_inv(state, hat_state)
%
% Inputs:
%    state - state
%    hat_state - state
%
% Outputs:
%    xi - uncertainty

% move to SE(2)
chi = [state.Rot state.p;
       0 0 1];
hat_chi = [hat_state.Rot hat_state.p;
       0 0 1];   
% compute log
xi = se2_log(se2_inv(chi) * hat_chi);
end