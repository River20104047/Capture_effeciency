clc, clear, close all

% This script compares two statistical models (Normal residuals and Negative Binomial)
% to fit a nonlinear function of the form:
%     c_i = e*D0 * sum_j a_ij * (1 - e)^(j-1)
% It estimates parameters D0 and e, computes diagnostics (AIC, BIC, R^2, RMSE),
% and generates 95% prediction intervals using Monte Carlo simulation.

% MATLAB script to fit two models: Normal (Model I) and Negative Binomial (Model II)

%% Input data
c = [27	45	34	19	31	22	17	29	37	26	22	9	13	25	14	34	22	23	19]';
W = [1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1]';

A = [3720.5	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
    3422.25	355	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
    2743.5	880.75	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
    2109.25	1204.5	574	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
    2389	821	509.25	151.5	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
    2167.5	901.25	728	212.75	83.25	0	0	0	0	0	0	0	0	0	0	0	0	0	0
    660.25	1235.75	967.75	861	235.5	83.25	0	0	0	0	0	0	0	0	0	0	0	0	0
    1751.5	1150.75	263.5	358.5	162.5	19.75	14.75	0	0	0	0	0	0	0	0	0	0	0	0
    996.5	1115.25	841.25	438	379.75	160	47.75	0	0	0	0	0	0	0	0	0	0	0	0
    950.5	752.25	1006.5	500.5	256	87	5.75	0	0	0	0	0	0	0	0	0	0	0	0
    709.5	1102.75	640	756.5	484.75	185.75	105	36.5	0	0	0	0	0	0	0	0	0	0	0
    356.75	727	854	814	535.75	226	162.75	30.25	0.5	0	0	0	0	0	0	0	0	0	0
    547	1049.5	660	440.75	528	445.25	126	85.5	19.25	0.5	0	0	0	0	0	0	0	0	0
    700.25	1032.5	1198	554.5	464.5	260.25	120.75	40	24.25	0	0	0	0	0	0	0	0	0	0
    427.25	724.75	868	640.25	506	400.5	304	141.75	45.75	11.5	0.5	0	0	0	0	0	0	0	0
    306.75	788	859.25	945.5	517	427.25	144.5	94.25	19	0	0	0	0	0	0	0	0	0	0
    1105.5	544.5	822.25	964.5	739.5	213.25	45.25	23.75	30.5	0.5	0	0	0	0	0	0	0	0	0
    86.5	371	487.25	702.5	768.25	737	406	206.5	149	50.25	4.5	0	0	0	0	0	0	0	0
    178	286.25	230	568.25	664.25	628.5	643.75	434.75	258.75	189.5	53.25	5	0	0	0	0	0	0	0];

n = length(c);
J = size(A, 2);

%% Helper function
% This function defines the model prediction given parameters:
% params(1) = D0 (scaling constant), params(2) = e (decay factor).
% It computes the sum over columns of matrix A weighted by (1 - e) powers.
model_fun = @(params, A) (params(1) * params(2)) * sum(A .* (1 - params(2)).^(0:J-1), 2);

%% === Model I: Normal residuals ===
% Assumes residuals follow a normal distribution.
% Uses weighted nonlinear least squares fitting (via lsqcurvefit).
% Also estimates standard errors and 95% CIs via Jacobian approximation.

% Apply weights to residuals (weighted least squares)
weighted_model_fun = @(params) sqrt(W) .* (model_fun(params, A));
weighted_c = sqrt(W) .* c;
params0 = [0.01, 0.1];
options = optimoptions('lsqcurvefit', 'Display', 'off');
[params_norm, resnorm, ~, ~, ~, ~, jacobian] = lsqcurvefit(@(p, ~) weighted_model_fun(p), params0, [], weighted_c, [0, 0], [Inf, 1], options);

D0_norm = params_norm(1);
e_norm = params_norm(2);
c_pred_norm = model_fun(params_norm, A);
residuals_norm = c - c_pred_norm;

% CI estimation using Jacobian
alpha = 0.05;
ci_norm = nlparci(params_norm, residuals_norm, 'jacobian', jacobian);
se_norm = (ci_norm(:,2) - ci_norm(:,1)) / (2 * norminv(0.975));

% Diagnostics
logL_norm = -n/2 * log(2*pi) - n/2 * log(resnorm/n) - resnorm/(2*resnorm/n);
AIC_norm = -2*logL_norm + 2*2;
BIC_norm = -2*logL_norm + log(n)*2;
R2_norm = 1 - sum(residuals_norm.^2) / sum((c - mean(c)).^2);
RMSE_norm = sqrt(mean(residuals_norm.^2));

%% === Model II: Negative Binomial ===
% Assumes observations follow a Negative Binomial distribution with overdispersion.
% Fits parameters D0, e, and r (dispersion) by maximizing weighted log-likelihood.
% Uses fmincon to estimate and inverts the Hessian for standard errors and CIs.

% Better initialization: use results from normal model as starting point
% and choose a reasonable dispersion parameter
D0_init = D0_norm;
e_init = e_norm;
r_init = 1.0;  % Start with moderate dispersion

% Improved likelihood function with better numerical stability
nb_likelihood = @(params) compute_nb_nll(params, c, A, W, model_fun);

params0_nb = [D0_init, e_init, r_init];
options_nb = optimoptions('fmincon','Display','off','Algorithm','interior-point',...
    'MaxIterations', 1000, 'MaxFunctionEvaluations', 3000, ...
    'StepTolerance', 1e-10, 'OptimalityTolerance', 1e-8);

% Set reasonable bounds
lb = [1e-6, 0, 1e-6];  % Lower bounds
ub = [Inf, 0.99, 100]; % Upper bounds

try
    [params_nb, nll_nb, exitflag, output, ~, ~, hessian_nb] = ...
        fmincon(nb_likelihood, params0_nb, [], [], [], [], lb, ub, [], options_nb);
    
    if exitflag <= 0
        warning('Optimization did not converge properly. Exit flag: %d', exitflag);
        fprintf('Optimization output: %s\n', output.message);
    end
    
    D0_nb = params_nb(1);
    e_nb = params_nb(2);
    r_nb = params_nb(3);
    c_pred_nb = model_fun(params_nb(1:2), A);
    
    % CI from inverse Hessian (with error checking)
    try
        cov_nb = inv(hessian_nb);
        se_nb = sqrt(diag(cov_nb));
        z = norminv(0.975);
        ci_nb = [params_nb' - z*se_nb, params_nb' + z*se_nb];
        hessian_success = true;
    catch
        warning('Could not invert Hessian matrix. Using approximate standard errors.');
        se_nb = [NaN; NaN; NaN];
        ci_nb = [NaN NaN; NaN NaN; NaN NaN];
        hessian_success = false;
    end
    
    % Diagnostics
    logL_nb = -nll_nb;
    AIC_nb = -2*logL_nb + 2*3;
    BIC_nb = -2*logL_nb + log(n)*3;
    R2_nb = 1 - sum((c - c_pred_nb).^2) / sum((c - mean(c)).^2);
    RMSE_nb = sqrt(mean((c - c_pred_nb).^2));
    
    nb_success = true;
    
catch ME
    warning('Negative Binomial model fitting failed: %s', E.message);
    % Set default values for failed fit
    D0_nb = NaN; e_nb = NaN; r_nb = NaN;
    c_pred_nb = NaN(size(c));
    se_nb = [NaN; NaN; NaN];
    ci_nb = [NaN NaN; NaN NaN; NaN NaN];
    logL_nb = NaN; AIC_nb = NaN; BIC_nb = NaN; R2_nb = NaN; RMSE_nb = NaN;
    nb_success = false;
    hessian_success = false;
end

%% === Output summary ===
fprintf('\nModel I (Normal residuals):\n');
fprintf('D0 = %.4f, e = %.4f\n', D0_norm, e_norm);
fprintf('95%% CI D0: [%.4f, %.4f], e: [%.4f, %.4f]\n', ci_norm(1,:), ci_norm(2,:));
fprintf('AIC = %.4f, BIC = %.4f, R^2 = %.4f, RMSE = %.4f\n', AIC_norm, BIC_norm, R2_norm, RMSE_norm);

if nb_success
    fprintf('\nModel II (Negative Binomial):\n');
    fprintf('D0 = %.4f, e = %.4f, r = %.4f\n', D0_nb, e_nb, r_nb);
    if hessian_success
        fprintf('95%% CI D0: [%.4f, %.4f], e: [%.4f, %.4f], r: [%.4f, %.4f]\n', ...
            ci_nb(1,1), ci_nb(1,2), ci_nb(2,1), ci_nb(2,2), ci_nb(3,1), ci_nb(3,2));
    else
        fprintf('95%% CI: Could not compute due to Hessian inversion failure\n');
    end
    fprintf('AIC = %.4f, BIC = %.4f, R^2 = %.4f, RMSE = %.4f\n', AIC_nb, BIC_nb, R2_nb, RMSE_nb);
else
    fprintf('\nModel II (Negative Binomial): FAILED TO FIT\n');
end

%% === Summary Table ===
if nb_success
    summary = table({'Normal'; 'NegBin'}, [AIC_norm; AIC_nb], [BIC_norm; BIC_nb], [R2_norm; R2_nb], [RMSE_norm; RMSE_nb],...
        'VariableNames', {'Model', 'AIC', 'BIC', 'R_squared', 'RMSE'});
else
    summary = table({'Normal'; 'NegBin'}, [AIC_norm; NaN], [BIC_norm; NaN], [R2_norm; NaN], [RMSE_norm; NaN],...
        'VariableNames', {'Model', 'AIC', 'BIC', 'R_squared', 'RMSE'});
end
disp('Model Comparison Summary:');
disp(summary);

% Enhanced parameter summary table including r parameter
param_summary = table({'Normal'; 'NegBin'},...
    [D0_norm; D0_nb], [se_norm(1); se_nb(1)], [ci_norm(1,1); ci_nb(1,1)], [ci_norm(1,2); ci_nb(1,2)],...
    [e_norm; e_nb], [se_norm(2); se_nb(2)], [ci_norm(2,1); ci_nb(2,1)], [ci_norm(2,2); ci_nb(2,2)],...
    [NaN; r_nb], [NaN; se_nb(3)], [NaN; ci_nb(3,1)], [NaN; ci_nb(3,2)],...
    'VariableNames', {'Model', 'D0', 'D0_SE', 'D0_CI_Low', 'D0_CI_High', 'e', 'e_SE', 'e_CI_Low', 'e_CI_High', 'r', 'r_SE', 'r_CI_Low', 'r_CI_High'});
disp('Parameter Estimates Summary (including r parameter):');
disp(param_summary);

if nb_success
    % Separate detailed summary for r parameter
    fprintf('\nDetailed r parameter results (Negative Binomial Model only):\n');
    fprintf('r = %.6f (SE = %.6f)\n', r_nb, se_nb(3));
    if hessian_success
        fprintf('95%% CI for r: [%.6f, %.6f]\n', ci_nb(3,1), ci_nb(3,2));
    else
        fprintf('95%% CI for r: Could not compute\n');
    end
    fprintf('Interpretation: r represents the dispersion parameter in the Negative Binomial model.\n');
    fprintf('Smaller r values indicate greater overdispersion relative to Poisson.\n');
end

%% === Monte Carlo CI for Predictions ===
% To estimate 95% predictive intervals, we sample (D0, e) from their estimated
% multivariate normal distribution and compute predicted values for each draw.
% Percentiles (2.5% and 97.5%) across simulations form the lower and upper bounds.

n_sim = 10000;

% --- Model I Monte Carlo ---
cov_norm = inv(jacobian' * jacobian);
param_samples_norm = mvnrnd(params_norm, cov_norm, n_sim);
c_pred_sim_norm = zeros(n_sim, n);

% Estimate residual variance for prediction interval
sigma_sq_norm = resnorm / (n - length(params_norm)); % resnorm is already sum of weighted squared residuals

for i = 1:n_sim
    D0_i = param_samples_norm(i,1);
    e_i = param_samples_norm(i,2);
    mean_pred = model_fun([D0_i, e_i], A)';
    
    % Add random noise from the estimated residual distribution
    c_pred_sim_norm(i,:) = mean_pred + randn(1, n) * sqrt(sigma_sq_norm); 
end
modelI_lower = prctile(c_pred_sim_norm, 2.5);
modelI_upper = prctile(c_pred_sim_norm, 97.5);

% --- Model II Monte Carlo ---
if nb_success && hessian_success
    % Sample all three parameters (D0, e, r)
    param_samples_nb_full = mvnrnd(params_nb, cov_nb, n_sim); 
    c_pred_sim_nb_obs = zeros(n_sim, n);

    for i = 1:n_sim
        D0_i = param_samples_nb_full(i,1);
        e_i = param_samples_nb_full(i,2);
        r_i = param_samples_nb_full(i,3); % Sample r

        % Ensure r_i is positive for nbinrnd
        if r_i <= 0
            r_i = 1e-6; % Small positive value or re-sample, depending on preferred handling
        end
        
        % Calculate the mean 'mu' for this set of parameters
        mu_i = model_fun([D0_i, e_i], A);
        
        % Ensure mu_i is positive for nbinrnd
        if any(mu_i <= 0)
            mu_i(mu_i <= 0) = 1e-6; % Ensure mu is positive
        end

        % Generate observations from Negative Binomial distribution
        % using nbinrnd(r, p) where p = r / (r + mu)
        p_i = r_i ./ (r_i + mu_i);
        
        % Ensure p_i is within (0,1) for nbinrnd
        p_i(p_i <= 0) = eps; % Small positive epsilon
        p_i(p_i >= 1) = 1 - eps; % Slightly less than 1
        
        c_pred_sim_nb_obs(i,:) = nbinrnd(r_i, p_i)'; 
    end
    modelII_lower = prctile(c_pred_sim_nb_obs, 2.5);
    modelII_upper = prctile(c_pred_sim_nb_obs, 97.5);
else
    modelII_lower = NaN(1, n);
    modelII_upper = NaN(1, n);
end

%% === Export to Excel ===
% Export parameter estimates, model diagnostics, and prediction results
% (including prediction intervals) to a timestamped Excel file.

% Ensure all vectors are column vectors for consistency
modelI_lower = modelI_lower(:);
modelI_upper = modelI_upper(:);
modelII_lower = modelII_lower(:);
modelII_upper = modelII_upper(:);
c_pred_norm = c_pred_norm(:);
if nb_success
    c_pred_nb = c_pred_nb(:);
else
    c_pred_nb = NaN(size(c));
end

% Sheet 1: Parameter Estimates
param_summary.Model = string(param_summary.Model);

% Sheet 2: Model Comparison
summary.Model = string(summary.Model);

% Sheet 3: Data vs Predictions
prediction_table = table((1:n)', c, c_pred_norm, modelI_lower, modelI_upper, c_pred_nb, modelII_lower, modelII_upper,...
    'VariableNames', {'Index', 'Observed', 'Pred_Model_I', 'Model_I_Lower', 'Model_I_Upper', 'Pred_Model_II', 'Model_II_Lower', 'Model_II_Upper'});

% Sheet 4: Detailed r parameter results
if nb_success
    r_results = table({'r_estimate'; 'r_SE'; 'r_CI_lower'; 'r_CI_upper'}, [r_nb; se_nb(3); ci_nb(3,1); ci_nb(3,2)],...
        'VariableNames', {'Parameter', 'Value'});
else
    r_results = table({'r_estimate'; 'r_SE'; 'r_CI_lower'; 'r_CI_upper'}, [NaN; NaN; NaN; NaN],...
        'VariableNames', {'Parameter', 'Value'});
end

% Write to Excel
timestamp = datestr(now, 'mmddHHMMSS');
filename = ['Fitting_results_' timestamp '.xlsx'];
try
    writecell({'Parameter Estimates (including r)'}, filename, 'Sheet', 1, 'Range', 'A1');
    writetable(param_summary, filename, 'Sheet', 1, 'Range', 'A2');
    
    writecell({'Model Comparison'}, filename, 'Sheet', 2, 'Range', 'A1');
    writetable(summary, filename, 'Sheet', 2, 'Range', 'A2');
    
    writecell({'Observed vs Predicted'}, filename, 'Sheet', 3, 'Range', 'A1');
    writetable(prediction_table, filename, 'Sheet', 3, 'Range', 'A2');
    
    writecell({'Detailed r Parameter Results (Negative Binomial Model)'}, filename, 'Sheet', 4, 'Range', 'A1');
    writetable(r_results, filename, 'Sheet', 4, 'Range', 'A2');
    
    fprintf('Results exported to: %s\n', filename);
catch
    warning('Could not export to Excel file');
end

%% === Visualization ===
% Plot observed data and predictions from both models for visual comparison.
figure;
hold on;
plot(1:n, c, 'ko-', 'DisplayName','Observed', 'LineWidth', 1, 'MarkerSize', 8);
plot(1:n, c_pred_norm, 'b--s', 'DisplayName','Model I (Normal)', 'LineWidth', 1, 'MarkerSize', 6);
if nb_success
    plot(1:n, c_pred_nb, 'r--d', 'DisplayName','Model II (NegBin)', 'LineWidth', 1, 'MarkerSize', 6);
end
legend('Location', 'best');
xlabel('Observation Index');
ylabel('c_i');
title('Observed vs Predicted (Model I and II)');
grid on;

% Add prediction intervals if available
if exist('modelI_lower', 'var') && ~any(isnan(modelI_lower))
    fill([1:n, fliplr(1:n)], [modelI_lower', fliplr(modelI_upper')], 'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'DisplayName', 'Model I 95% PI');
end
if nb_success && hessian_success && ~any(isnan(modelII_lower))
    fill([1:n, fliplr(1:n)], [modelII_lower', fliplr(modelII_upper')], 'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'DisplayName', 'Model II 95% PI');
end

%% Helper function for Negative Binomial likelihood
function nll = compute_nb_nll(params, c, A, W, model_fun)
    % Compute negative log-likelihood for Negative Binomial model
    % with better numerical stability
    
    D0 = params(1);
    e = params(2);
    r = params(3);
    
    % Check parameter bounds
    if D0 <= 0 || e <= 0 || e >= 1 || r <= 0
        nll = Inf;
        return;
    end
    
    % Compute predicted means
    mu = model_fun([D0, e], A);
    
    % Check for valid predictions
    if any(mu <= 0) || any(~isfinite(mu))
        nll = Inf;
        return;
    end
    
    % Negative Binomial parameters
    % Using parameterization where p = r/(r + mu)
    p = r ./ (r + mu);
    
    % Check for valid probabilities
    if any(p <= 0) || any(p >= 1) || any(~isfinite(p))
        nll = Inf;
        return;
    end
    
    % Compute log-likelihood components
    try
        % Round observations to integers for nbinpdf
        c_round = round(c);
        
        % Compute log probabilities
        log_probs = zeros(size(c));
        for i = 1:length(c)
            if c_round(i) >= 0
                log_probs(i) = log(nbinpdf(c_round(i), r, p(i)) + eps);
            else
                log_probs(i) = -Inf;
            end
        end
        
        % Weighted negative log-likelihood
        nll = -sum(W .* log_probs);
        
        % Check for valid result
        if ~isfinite(nll)
            nll = Inf;
        end
        
    catch
        nll = Inf;
    end
end
