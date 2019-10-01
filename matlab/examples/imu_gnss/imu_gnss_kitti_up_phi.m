function [new_state] = imu_gnss_kitti_up_phi(state, xi)
%IMU_GNSS_KITTI_UP_PHI retraction
%
% Syntax: [new_state] = imu_gnss_kitti_up_phi(state, xi)
%
% Inputs:
%    state - state
%    xi - uncertainty
%
% Outputs:
%    new_state - state

new_state = state;
new_state.p = state.p + xi(1:3);
end
