
f = @(x) squeeze(mean(x, 2));
left_ukf_nees = f(left_ukf_nees);
right_ukf_nees = f(right_ukf_nees);
ekf_nees = f(ekf_nees);

fig2 = figure;
set(gca, 'YScale', 'log')
title('Robot position NEES')
xlabel('$t$ (s)')
ylabel('position NEES')
hold on;
grid on;
plot(t, left_ukf_nees, 'g');
plot(t, right_ukf_nees, 'c');
plot(t, ekf_nees, 'r');
legend('\textbf{left UKF}', '\textbf{right UKF}', 'EKF');

f = @(x) sprintf('%0.2f', mean(x));
left_ukf_nees = f(left_ukf_nees);
right_ukf_nees = f(right_ukf_nees);
ekf_nees = f(ekf_nees);

if cur_folder == "matlab"
    print(fig2, 'benchmarks/html/figures/benchmark_attitude_02', ...
        '-dpng', '-r600')
end
disp(' ')
disp('Normalized Estimation Error Squared (NEES)');
disp("    -left UKF : " + left_ukf_nees);
disp("    -right UKF: " + right_ukf_nees);
disp("    -EKF      : " + ekf_nees);
