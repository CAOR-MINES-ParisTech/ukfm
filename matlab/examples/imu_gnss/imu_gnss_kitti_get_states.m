function [Rots, vs, ps, b_gyros, b_accs] = imu_gnss_kitti_get_states(states)
%IMU_GNSS_KITTI_GET_STATES
%
% Syntax: [Rots, vs, ps] = imu_gnss_kitti_get_states(states)
%
% Inputs:
%    states - states
% Outputs:
%    Rots - orientation matrices
%    vs - velocities
%    ps - positions
%    b_gyros - gyro biases
%    b_accs - accelerometer biases

state_table = struct2table(states);
Rots = state_table.Rot;
vs = cell2mat(state_table.v')';
ps = cell2mat(state_table.p')';
b_gyros = cell2mat(state_table.b_gyro')';
b_accs = cell2mat(state_table.b_acc')';
end

