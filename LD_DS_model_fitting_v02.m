%% This is used to fit LD and DeLury models

% Prepare workspace
clc, clear, close all

% Data
c = [27	45	34	19	31	22	17	29	37	26	22	9	13	25	14	34	22	23	19]';
f = [3720.5	3777.25	3624.25	3887.75	3870.75	4092.75	4043.5	3721.25	3978.5	3558.5	4020.75	3707	3901.75	4395	4070.25	4101.5	4489.5	3968.75	4140.25]';
% Weight vector and diagonal matrix
w = [1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1]';

W_mat = diag(w);

% Derived values
cpue = c ./  f;
K = [0; cumsum(c(1:end-1))];   % Cumulative catch (Leslie-Davis)
E = [0; cumsum(f(1:end-1))];   % Cumulative effort (DeLury)
log_cpue = log(cpue);

% ===== Weighted Regression: Leslie-Davis =====
X_LD = [ones(size(K)), K];
y_LD = cpue;
b_LD = (X_LD' * W_mat * X_LD) \ (X_LD' * W_mat * y_LD);
yhat_LD = X_LD * b_LD;
res_LD = y_LD - yhat_LD;

% ===== Weighted Regression: DeLury =====
X_DL = [ones(size(E)), E];
y_DL = log_cpue;
b_DL = (X_DL' * W_mat * X_DL) \ (X_DL' * W_mat * y_DL);
yhat_DL = X_DL * b_DL;
res_DL = y_DL - yhat_DL;

% Metrics
n = length(c);
mdl_metrics = @(y, yhat, k) struct( ...
    'R2', 1 - sum((y - yhat).^2) / sum((y - mean(y)).^2), ...
    'RMSE', sqrt(mean((y - yhat).^2)), ...
    'AIC', n*log(sum((y - yhat).^2)/n) + 2*k, ...
    'BIC', n*log(sum((y - yhat).^2)/n) + k*log(n));

metrics_LD = mdl_metrics(cpue, yhat_LD, 2);
metrics_DL = mdl_metrics(log_cpue, yhat_DL, 2);

% Standard errors and CIs
se_LD = sqrt(diag(inv(X_LD' * X_LD) * var(res_LD)));
se_DL = sqrt(diag(inv(X_DL' * X_DL) * var(res_DL)));
CI95 = @(b, se) [b - 1.96*se, b + 1.96*se];

% Estimates
q_LD = -b_LD(2);
N0_LD = b_LD(1) / q_LD;

q_DL = -b_DL(2);
N0_DL = exp(b_DL(1)) / q_DL;

% === FIGURE 1: Observed vs Predicted CPUE (one plot) ===
figure(1); clf;
plot(1:n, cpue, 'ko-', 'LineWidth', 1.5, 'DisplayName', 'Observed CPUE'); hold on;
plot(1:n, yhat_LD, 'r*-', 'LineWidth', 1.5, 'DisplayName', 'Predicted (Leslie-Davis)');
plot(1:n, exp(yhat_DL), 'bs-', 'LineWidth', 1.5, 'DisplayName', 'Predicted (DeLury)');
xlabel('Effort Index (t)');
ylabel('CPUE');
title('Observed vs Predicted CPUE');
legend('Location', 'best'); grid on;

% === FIGURE 2: Linear Regression Data ===
figure(2); clf;

subplot(1,2,1);
scatter(K, cpue, 60, 'ko', 'filled'); hold on;
plot(K, yhat_LD, 'r-', 'LineWidth', 2);
xlabel('Cumulative Catch (K_{t-1})');
ylabel('CPUE');
title('Leslie-Davis Regression');
grid on;

subplot(1,2,2);
scatter(E, log_cpue, 60, 'ko', 'filled'); hold on;
plot(E, yhat_DL, 'b-', 'LineWidth', 2);
xlabel('Cumulative Effort (E_{t-1})');
ylabel('log(CPUE)');
title('DeLury Regression');
grid on;

% (Optional) Display summary tables
summary_table = table( ...
    ["Leslie-Davis"; "DeLury"], ...
    [N0_LD; N0_DL], ...
    [q_LD; q_DL], ...
    [se_LD(1)/q_LD; exp(b_DL(1))*se_DL(1)/(q_DL*exp(b_DL(1)))], ...
    [se_LD(2); se_DL(2)], ...
    [CI95(N0_LD, se_LD(1)/q_LD); CI95(N0_DL, exp(b_DL(1))*se_DL(1)/(q_DL*exp(b_DL(1))))], ...
    [CI95(q_LD, se_LD(2)); CI95(q_DL, se_DL(2))], ...
    [metrics_LD.R2; metrics_DL.R2], ...
    [metrics_LD.RMSE; metrics_DL.RMSE], ...
    [metrics_LD.AIC; metrics_DL.AIC], ...
    [metrics_DL.BIC; metrics_DL.BIC], ...
    'VariableNames', {'Model','N0','q','SE_N0','SE_q','CI95_N0','CI95_q','R2','RMSE','AIC','BIC'});

disp('Summary Table:');
disp(summary_table);

%% Back calculation
% === BACK-CALCULATE predicted catch ===
c_hat_LD = yhat_LD .* f;
c_hat_DL = exp(yhat_DL) .* f;

%% Summary table
% === DERIVED VALUES TABLE ===
derived_table = table((1:n)', c, f, cpue, K, E, log_cpue, yhat_LD, exp(yhat_DL), ...
    c_hat_LD, c_hat_DL, ...
    'VariableNames', {'t','Catch','Effort','CPUE','K_t_minus1','E_t_minus1','log_CPUE', ...
                      'Pred_CPUE_LD','Pred_CPUE_DL','Pred_Catch_LD','Pred_Catch_DL'});

disp('Derived Values Table:');
disp(derived_table);

%% Export to Excel

% === MONTE CARLO SIMULATION for 95% CI of predicted catch ===
rng(1); % For reproducibility
n_sim = 10000;

% ---- Leslie-Davis ----
cov_LD = inv(X_LD' * X_LD) * var(res_LD); % Covariance of LD params
b_samples_LD = mvnrnd(b_LD', cov_LD, n_sim); % [10000 x 2]

c_sim_LD = zeros(n_sim, n); % Initialize for simulated catches
std_dev_res_LD = sqrt(var(res_LD)); % Standard deviation of residuals for Leslie-Davis

for i = 1:n_sim
    % Predicted mean CPUE for sampled parameters
    cpue_pred_mean_i = b_samples_LD(i,1) + b_samples_LD(i,2) .* K'; 
    
    % Add normal random noise to CPUE prediction to account for residual variability
    % Assuming normally distributed errors with constant variance
    cpue_sim_obs_i = cpue_pred_mean_i + randn(1, n) * std_dev_res_LD; 
    
    % Convert to catch
    c_sim_LD(i,:) = cpue_sim_obs_i .* f'; 
end

ci_c_LD = prctile(c_sim_LD, [2.5 97.5])'; % [n x 2] transposed to match time

% ---- DeLury ----
cov_DL = inv(X_DL' * X_DL) * var(res_DL); % Covariance of DL params
b_samples_DL = mvnrnd(b_DL', cov_DL, n_sim); % [10000 x 2]

c_sim_DL = zeros(n_sim, n); % Initialize for simulated catches
std_dev_res_DL = sqrt(var(res_DL)); % Standard deviation of residuals for DeLury (on log scale)

for i = 1:n_sim
    % Predicted mean log(CPUE) for sampled parameters
    log_cpue_pred_mean_i = b_samples_DL(i,1) + b_samples_DL(i,2) .* E'; 
    
    % Add normal random noise to log(CPUE) prediction
    % Assuming normally distributed errors with constant variance on the log scale
    log_cpue_sim_obs_i = log_cpue_pred_mean_i + randn(1, n) * std_dev_res_DL; 
    
    % Exponentiate to get CPUE, then convert to catch
    c_sim_DL(i,:) = exp(log_cpue_sim_obs_i) .* f';
end

ci_c_DL = prctile(c_sim_DL, [2.5 97.5])'; % [n x 2]


% === Export to Excel ===

% Generate timestamped filename
timestamp = datestr(now, 'mmdd-HHMMSS');
filename = ['A04_Results_' timestamp '.xlsx'];

% ---------------- Sheet 1: Summary Table ----------------
writetable(summary_table, filename, 'Sheet', 'Model_Summary', 'WriteRowNames', false);

% ---------------- Sheet 2: Derived Values ----------------
writetable(derived_table, filename, 'Sheet', 'Derived_Values', 'WriteRowNames', false);

% ---------------- Sheet 3: Figure 1 Data ----------------
fig1_table = table((1:n)', cpue, yhat_LD, exp(yhat_DL), ...
    'VariableNames', {'Effort_Index','Observed_CPUE','Pred_CPUE_LD','Pred_CPUE_DL'});
writetable(fig1_table, filename, 'Sheet', 'Figure1_CPUE', 'WriteRowNames', false);

% ---------------- Sheet 4: Figure 2 Data ----------------
fig2_table_LD = table(K, cpue, yhat_LD, ...
    'VariableNames', {'Cumulative_Catch_K', 'Observed_CPUE', 'Pred_CPUE_LD'});
fig2_table_DL = table(E, log_cpue, yhat_DL, ...
    'VariableNames', {'Cumulative_Effort_E', 'Observed_log_CPUE', 'Pred_log_CPUE_DL'});

% Write both regression input/output to the same sheet
writetable(fig2_table_LD, filename, 'Sheet', 'Figure2_LeslieDavis', 'WriteRowNames', false);
writetable(fig2_table_DL, filename, 'Sheet', 'Figure2_DeLury', 'WriteRowNames', false);

% === Update Figure 3 Table with 95% CI ===
fig3_table = table((1:n)', c, c_hat_LD, ci_c_LD(:,1), ci_c_LD(:,2), ...
                          c_hat_DL, ci_c_DL(:,1), ci_c_DL(:,2), ...
    'VariableNames', {'Effort_Index','Observed_Catch', ...
                      'Pred_Catch_LD','CI_LD_Low','CI_LD_High', ...
                      'Pred_Catch_DL','CI_DL_Low','CI_DL_High'});

% Export to Excel
writetable(fig3_table, filename, 'Sheet', 'Figure3_Catch', 'WriteRowNames', false);
% Confirm
fprintf('✅ Data exported to Excel file: %s\n', filename);

% === FIGURE 3: Observed vs Predicted Catch ===
figure(3); clf;
hold on; % Keep the plot active for adding more elements

plot(1:n, c, 'ko-', 'LineWidth', 1, 'DisplayName', 'Observed Catch');
plot(1:n, c_hat_LD, 'r*-', 'LineWidth', 1, 'DisplayName', 'Predicted (Leslie-Davis)');
plot(1:n, c_hat_DL, 'bs-', 'LineWidth', 1, 'DisplayName', 'Predicted (DeLury)');

% Add prediction intervals for Leslie-Davis
% Corrected line: Use all(~isnan(ci_c_LD(:))) to ensure a single logical scalar
if exist('ci_c_LD', 'var') && all(~isnan(ci_c_LD(:)))
    % Reshape ci_c_LD for fill function (needs a single row or column per bound)
    % ci_c_LD is [n x 2], so ci_c_LD(:,1)' is lower, ci_c_LD(:,2)' is upper
    fill([1:n, fliplr(1:n)], [ci_c_LD(:,1)', fliplr(ci_c_LD(:,2)')], 'r', ...
         'FaceAlpha', 0.2, 'EdgeColor', 'none', 'DisplayName', 'Leslie-Davis 95% PI');
end

% Add prediction intervals for DeLury
% Corrected line: Use all(~isnan(ci_c_DL(:))) to ensure a single logical scalar
if exist('ci_c_DL', 'var') && all(~isnan(ci_c_DL(:)))
    fill([1:n, fliplr(1:n)], [ci_c_DL(:,1)', fliplr(ci_c_DL(:,2)')], 'b', ...
         'FaceAlpha', 0.2, 'EdgeColor', 'none', 'DisplayName', 'DeLury 95% PI');
end

xlabel('Effort Index (t)');
ylabel('Catch (c)');
title('Observed vs Predicted Catch with 95% Prediction Intervals');
legend('Location', 'best');
grid on;
hold off; % Release the plot