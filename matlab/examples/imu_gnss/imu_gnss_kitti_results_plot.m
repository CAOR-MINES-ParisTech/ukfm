function [] = imu_gnss_kitti_results_plot(ukf_states, ys)
%IMU_GNSS_KITTI_RESULTS_PLOT plot
%
% Syntax: [] = imu_gnss_kitti_results_plot(ukf_states, ys)

set(groot,'defaulttextinterprete','latex');  
set(groot, 'defaultAxesTickLabelInterprete','latex');  
set(groot, 'defaultLegendInterprete','latex');


% get estimated state
ukf_state_table = struct2table(ukf_states);
ukf_Rots = ukf_state_table.Rot;
ukf_ps = cell2mat(ukf_state_table.p')';


fig = figure;
title('Position in $xy$-plan')
xlabel('$x$ (m)')
ylabel('$y$ (m)')
hold on;
grid on;
scatter(ys(1, :), ys(2, :), 'r');
plot(ukf_ps(:, 1), ukf_ps(:, 2), 'b');
legend('GPS measurements', 'UKF');
axis equal;
end

