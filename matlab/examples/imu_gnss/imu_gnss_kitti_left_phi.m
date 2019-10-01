function [new_state] = imu_gnss_kitti_left_phi(state, xi)
%IMU_GNSS_KITTI_LEFT_PHI retraction
%
% Syntax: [new_state] = imu_gnss_kitti_left_phi(state, xi)
%
% Inputs:
%    state - state
%    xi - uncertainty
%
% Outputs:
%    state - state

dR = so3_exp(xi);
J = so3_left_jacobian(xi(1:3));
new_state.Rot = state.Rot - dR;
new_state.v = new_state.Rot * J*xi(4:6) + state.v;
new_state.p = new_state.Rot * J*xi(7:9) + state.p;
new_state.b_gyro = state.b_gyro + xi(10:12);
new_state.b_acc = state.b_acc + xi(13:15);
end