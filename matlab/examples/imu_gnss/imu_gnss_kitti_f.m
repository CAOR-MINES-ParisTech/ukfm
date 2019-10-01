function [new_state] = imu_gnss_kitti_f(state, omega, w, dt)
%IMU_GNSS_KITTI_F Propagate state
%
% Syntax:  [new_state] = imu_gnss_kitti_f(state, omega, w, dt)
%
% Inputs:
%    state - state
%    omega - input
%    w - input noise
%    dt - integration step
%
% Outputs:
%    new_state - propagated state

g = [0; 0; 9.82];
gyro = omega.gyro - state.b_gyro + w(1:3);
acc = state.Rot*(omega.acc - state.b_acc + w(4:6)) - g;
new_state.Rot = state.Rot * so3_exp(gyro*dt);
new_state.v = state.v + acc*dt;
new_state.p = state.p + state.v*dt + 1/2*acc*dt^2;
% noise is not added on bias
new_state.b_gyro = state.b_gyro; % + w(7:9)*dt;
new_state.b_acc = state.b_acc; % + w(10:12)*dt;
end

