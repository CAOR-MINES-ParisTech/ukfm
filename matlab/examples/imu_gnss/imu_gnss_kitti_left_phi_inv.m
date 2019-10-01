function [xi] = imu_gnss_kitti_left_phi_inv(state, hat_state)
%IMU_GNSS_KITTI_LEFT_PHI_INV inverse retraction
%
% Syntax: [xi] = imu_gnss_kitti_left_phi_inv(state, hat_state)
%
% Inputs:
%    state - state
%    hat_state - state
%    xi - uncertainty
%
% Outputs:
%    xi - uncertainty

dR = state.Rot' * hat_state.Rot;
phi = so3_log(dR);
J = so3_inv_left_jacobian(phi);
dv = state.Rot'*(hat_state.v - state.v);
dp = state.Rot'*(hat_state.p - state.p);
xi = [phi;
    J*dv;
    J*dp;
    hat_state.b_gyro - state.b_gyro;
    hat_state.b_acc - state.b_acc];
end