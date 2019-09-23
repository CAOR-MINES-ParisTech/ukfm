function [xi] = imu_gnss_kitti_right_phi_inv(state, hat_state)
%imu_gnss_KITTI_RIGHT_PHI_INV inverse retraction
%
% Syntax: [xi] = imu_gnss_kitti_right_phi_inv(state, hat_state)
%
% Inputs:
%    state - state
%    hat_state - state
%    xi - uncertainty
%
% Outputs:
%    xi - uncertainty

dR = hat_state.Rot * state.Rot';
phi = so3_log(dR);
J = so3_inv_left_jacobian(phi);
dv = hat_state.v - dR*state.v;
dp = hat_state.p - dR*state.p;
xi = [phi;
    J*dv;
    J*dp;
    hat_state.b_gyro - state.b_gyro;
    hat_state.b_acc - state.b_acc];
end