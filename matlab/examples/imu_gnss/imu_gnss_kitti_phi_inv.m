function [xi] = imu_gnss_kitti_phi_inv(state, hat_state)
%IMU_GNSS_KITTI_PHI_INV inverse retraction
%
% Syntax: [xi] = imu_gnss_kitti_phi_inv(state, hat_state)
%
% Inputs:
%    state - state
%    hat_state - state
%
% Outputs:
%    xi - uncertainty

xi = [so3_log(hat_state.Rot * state.Rot');
    hat_state.v - state.v;
    hat_state.p - state.p;
    hat_state.b_gyro - state.b_gyro;
    hat_state.b_acc - state.b_acc];
end

