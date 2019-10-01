function [] = imu_gnss_kitti_results_plot(ukf_states, ys)
%IMU_GNSS_KITTI_RESULTS_PLOT
%
% Syntax: [] = imu_gnss_kitti_results_plot(ukf_states, ys)

set(groot,'defaulttextinterprete','latex');  
set(groot, 'defaultAxesTickLabelInterprete','latex');  
set(groot, 'defaultLegendInterprete','latex');
cur_folder = pwd;
cur_folder = cur_folder(end-5:end);

% get estimated state
[~, ~, ukf_ps, ~, ~] = imu_gnss_kitti_get_states(ukf_states);


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

if cur_folder == "matlab"
    print(fig, 'examples/html/figures/main_imugnss', '-dpng', '-r600')
end
end

