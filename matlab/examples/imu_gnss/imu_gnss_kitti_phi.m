function [new_state] = imu_gnss_kitti_phi(state, xi)
%IMU_GNSS_KITTI_PHI retraction
%
% Syntax: [new_state] = imu_gnss_kitti_phi(state, xi)
%
% Inputs:
%    state - state
%    xi - uncertainty
%
% Outputs:
%    new_state - state

new_state.Rot = so3_exp(xi(1:3)) * state.Rot;
new_state.v = state.v + xi(4:6);
new_state.p = state.p + xi(7:9);
new_state.b_gyro = state.b_gyro + xi(10:12);
new_state.b_acc = state.b_acc + xi(13:15);
end
