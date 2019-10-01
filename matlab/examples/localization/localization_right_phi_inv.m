function [xi] = localization_right_phi_inv(state, hat_state)
%LOCALIZATION_RIGHT_PHI_INV inverse retraction
%
% Syntax: [xi] = localization_right_phi_inv(state, hat_state)
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
xi = se2_log(hat_chi * se2_inv(chi));
end
