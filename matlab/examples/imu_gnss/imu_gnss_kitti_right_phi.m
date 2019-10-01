function [new_state] = imu_gnss_kitti_right_phi(state, xi)
%IMU_GNSS_KITTI_PHI retraction
%
% Syntax: [new_state] = imu_gnss_kitti_right_phi(state, xi)
%
% Inputs:
%    state - state
%    xi - uncertainty
%
% Outputs:
%    state - state

dR = so3_exp(xi);
J = so3_left_jacobian(xi(1:3));
new_state.Rot = dR * state.Rot;
new_state.v = dR * state.v + J*xi(4:6);
new_state.p = dR * state.p + J*xi(7:9);
new_state.b_gyro = state.b_gyro + xi(10:12);
new_state.b_acc = state.b_acc + xi(13:15);
end