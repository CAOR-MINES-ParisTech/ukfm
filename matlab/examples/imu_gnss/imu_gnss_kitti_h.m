function [y] = imu_gnss_kitti_h(state)
%imu_gnss_KITTI_H Measurement
%
% Syntax: [y] = imu_gnss_kitti_h(state)
%
% Inputs.
%    state - state
%
% Outputs:
%    y - measurement

y = state.p;
end