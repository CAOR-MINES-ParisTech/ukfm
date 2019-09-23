% get estimates and plot figure
state_table = struct2table(true_state);
ukf_left_state_table = struct2table(ukf_left_states);
ukf_right_state_table = struct2table(ukf_right_states);
ekf_state_table = struct2table(ekf_states);

ukf_left_err_rot = sprintf('%0.2f', sqrt(mean(ukf_left_err(1, :))/N));
ukf_right_err_rot = sprintf('%0.2f', sqrt(mean(ukf_right_err(1, :))/N));
ekf_err_rot = sprintf('%0.2f', sqrt(mean(ekf_err(1, :))/N));


