%% 4.1. MATLAB Code
% The MATLAB code used for this analysis is divided into two parts.

% 4.1.1. Part A.m
% Code for data loading, analysis, and model building.

%% Step 1 Load the Dataset
% Load the dataset from the Excel file into a MATLAB table
data = readtable(Concrete_Data.xls, 'VariableNamingRule', 'preserve');

% Display the first few rows of the dataset to verify it loaded correctly
disp('First few rows of the dataset');
head(data);

%% Step 2 Calculate Basic Statistics
% Create a table to store basic statistics (max, min, mean, std) for each variable
stats = table();
stats.Variable = data.Properties.VariableNames; % Store variable names
stats.Max = max(data{,}); % Calculate maximum values
stats.Min = min(data{,}); % Calculate minimum values
stats.Mean = mean(data{,}); % Calculate mean values
stats.Std = std(data{,}); % Calculate standard deviation

% Display the basic statistics
disp('Basic Statistics');
disp(stats);

% Export the statistics to an Excel file
writetable(stats, 'statistics.xlsx');

%% Step 3 Calculate Ratios
% Create a table to store calculated ratios
ratios = table();

% Calculate the water-to-cement ratio
ratios.WaterToCement = data.(Water  (component 4)(kg in a m^3 mixture)) . data.(Cement (component 1)(kg in a m^3 mixture));

% Calculate the water-to-binder ratio (binder = cement + fly ash + slag)
ratios.WaterToBinder = data.(Water  (component 4)(kg in a m^3 mixture)) . ...
    (data.(Cement (component 1)(kg in a m^3 mixture)) + ...
    data.(Fly Ash (component 3)(kg in a m^3 mixture)) + ...
    data.(Blast Furnace Slag (component 2)(kg in a m^3 mixture)));

% Calculate the fly ash-to-binder ratio
ratios.FlyAshToBinder = data.(Fly Ash (component 3)(kg in a m^3 mixture)) . ...
    (data.(Cement (component 1)(kg in a m^3 mixture)) + ...
    data.(Fly Ash (component 3)(kg in a m^3 mixture)) + ...
    data.(Blast Furnace Slag (component 2)(kg in a m^3 mixture)));

% Calculate the slag-to-binder ratio
ratios.SlagToBinder = data.(Blast Furnace Slag (component 2)(kg in a m^3 mixture)) . ...
    (data.(Cement (component 1)(kg in a m^3 mixture)) + ...
    data.(Fly Ash (component 3)(kg in a m^3 mixture)) + ...
    data.(Blast Furnace Slag (component 2)(kg in a m^3 mixture)));

% Calculate the (fly ash + slag)-to-binder ratio
ratios.FlyAshAndSlagToBinder = (data.(Fly Ash (component 3)(kg in a m^3 mixture)) + ...
    data.(Blast Furnace Slag (component 2)(kg in a m^3 mixture))) . ...
    (data.(Cement (component 1)(kg in a m^3 mixture)) + ...
    data.(Fly Ash (component 3)(kg in a m^3 mixture)) + ...
    data.(Blast Furnace Slag (component 2)(kg in a m^3 mixture)));

% Export the ratios to a new sheet in the Excel file
writetable(ratios, 'statistics.xlsx', 'Sheet', 'Ratios');

%% Step 4 Visualize Relationships
% Create scatter plots to visualize the relationship between each variable and compressive strength
figure;
for i = 1width(data)-1
    subplot(3, 3, i); % Create subplots in a 3x3 grid
    scatter(data{, i}, data.(Concrete compressive strength(MPa, megapascals))); % Scatter plot
    mean_val = mean(data{, i}); % Calculate mean of the variable
    std_val = std(data{, i}); % Calculate standard deviation of the variable
    title(sprintf('%snMean %.2f, Std %.2f', data.Properties.VariableNames{i}, mean_val, std_val)); % Add title
    xlabel(data.Properties.VariableNames{i}); % Label x-axis
    ylabel('Compressive Strength (MPa)'); % Label y-axis
    grid on; % Add gridlines
    box on;  % Add box borders
end
% Save the figure as a JPG file
saveas(gcf, 'scatter_plots_with_stats.jpg');

%% Step 4.2 Visualize Relationships for Ratios
% Create scatter plots for the calculated ratios and compressive strength
figure;
subplot(2, 3, 1);
scatter(ratios.WaterToCement, data.(Concrete compressive strength(MPa, megapascals)));
title(sprintf('Water-to-CementnMean %.2f, Std %.2f', mean(ratios.WaterToCement), std(ratios.WaterToCement)));
xlabel('Water-to-Cement Ratio');
ylabel('Compressive Strength (MPa)');
grid on;
box on;

subplot(2, 3, 2);
scatter(ratios.WaterToBinder, data.(Concrete compressive strength(MPa, megapascals)));
title(sprintf('Water-to-BindernMean %.2f, Std %.2f', mean(ratios.WaterToBinder), std(ratios.WaterToBinder)));
xlabel('Water-to-Binder Ratio');
ylabel('Compressive Strength (MPa)');
grid on;
box on;

subplot(2, 3, 3);
scatter(ratios.FlyAshToBinder, data.(Concrete compressive strength(MPa, megapascals)));
title(sprintf('Fly Ash-to-BindernMean %.2f, Std %.2f', mean(ratios.FlyAshToBinder), std(ratios.FlyAshToBinder)));
xlabel('Fly Ash-to-Binder Ratio');
ylabel('Compressive Strength (MPa)');
grid on;
box on;

subplot(2, 3, 4);
scatter(ratios.SlagToBinder, data.(Concrete compressive strength(MPa, megapascals)));
title(sprintf('Slag-to-BindernMean %.2f, Std %.2f', mean(ratios.SlagToBinder), std(ratios.SlagToBinder)));
xlabel('Slag-to-Binder Ratio');
ylabel('Compressive Strength (MPa)');
grid on;
box on;

subplot(2, 3, 5);
scatter(ratios.FlyAshAndSlagToBinder, data.(Concrete compressive strength(MPa, megapascals)));
title(sprintf('Fly Ash + Slag-to-BindernMean %.2f, Std %.2f', mean(ratios.FlyAshAndSlagToBinder), std(ratios.FlyAshAndSlagToBinder)));
xlabel('Fly Ash + Slag-to-Binder Ratio');
ylabel('Compressive Strength (MPa)');
grid on;
box on;

% Save the figure as a JPG file
saveas(gcf, 'scatter_plots_for_ratios.jpg');

%% Step 5 Calculate Correlation Coefficients
% Extract input variables (all columns except the last one) and output variable (last column)
input_vars = data{, 1end-1}; % Input variables
output_var = data{, end};     % Output variable (compressive strength)

% Calculate correlation coefficients between input variables and compressive strength
correlation_matrix = corrcoef([input_vars, output_var]);

% Extract the correlation coefficients between input variables and compressive strength
correlation_with_strength = correlation_matrix(1end-1, end);

% Display the correlation coefficients in a table
correlation_table = array2table(correlation_with_strength, ...
    'VariableNames', {'Correlation'}, ...
    'RowNames', data.Properties.VariableNames(1end-1));
disp('Correlation Coefficients');
disp(correlation_table);

%% Step 6 Identify Significant Variables
% Prompt the user to input a correlation threshold
threshold = input('Enter the correlation coefficient threshold (e.g., 0.5) ');

% Find variables with correlation coefficients above the threshold
significant_vars = correlation_with_strength(abs(correlation_with_strength)  threshold);

% Get the names of the significant variables
significant_variable_names = data.Properties.VariableNames(abs(correlation_with_strength)  threshold);

% Display the significant variables
disp('Significant Variables');
disp(significant_variable_names');

%% Step 6.2 3D Scatter Plots for Top Two Influential Variables
% Check if there are at least two significant variables
if length(significant_variable_names) = 2
    % Extract the top two influential variables
    top_two_vars = significant_variable_names(12);
    
    % Extract the data for the top two variables and compressive strength
    var1 = data{, top_two_vars{1}};
    var2 = data{, top_two_vars{2}};
    strength = data.(Concrete compressive strength(MPa, megapascals));
    
    % Create a 3D scatter plot
    figure;
    scatter3(var1, var2, strength, 'filled');
    xlabel(top_two_vars{1});
    ylabel(top_two_vars{2});
    zlabel('Compressive Strength (MPa)');
    title(sprintf('3D Scatter Plot %s vs %s vs Compressive Strength', top_two_vars{1}, top_two_vars{2}));
    grid on;
    box on;
    
    % Save the plot as a JPG file
    saveas(gcf, '3d_scatter_plot_top_two_variables.jpg');
else
    disp('Not enough significant variables to create a 3D scatter plot.');
end

%% Step 7 Linear Regression
% Extract the significant variables and the output variable
significant_data = data{, significant_variable_names}; % Input variables
compressive_strength = data.(Concrete compressive strength(MPa, megapascals)); % Output variable

% Split the data into training and testing sets (80% training, 20% testing)
rng(42); % Set a random seed for reproducibility
split_ratio = 0.8; % 80% training, 20% testing
split_index = floor(split_ratio  height(data)); % Index to split the data

% Training data
X_train = significant_data(1split_index, );
y_train = compressive_strength(1split_index);

% Testing data
X_test = significant_data(split_index+1end, );
y_test = compressive_strength(split_index+1end);

% Fit a linear regression model using the training data
linear_model = fitlm(X_train, y_train);

% Display the model summary
disp('Linear Regression Model Summary');
disp(linear_model);

% Predict compressive strength on the testing data
y_pred = predict(linear_model, X_test);

% Calculate evaluation metrics R², RMSE, and MAE
ss_total_linear = sum((y_test - mean(y_test)).^2);       % Total sum of squares
ss_residual_linear = sum((y_test - y_pred).^2);         % Residual sum of squares
r2_score_linear = 1 - (ss_residual_linear  ss_total_linear);   % R² score
rmse = sqrt(mean((y_test - y_pred).^2));  % Root Mean Squared Error
mae = mean(abs(y_test - y_pred));         % Mean Absolute Error

% Display the evaluation metrics
disp('Linear Model Performance Metrics (Testing Data)');
fprintf('R² Score %.4fn', r2_score_linear);
fprintf('RMSE %.4fn', rmse);
fprintf('MAE %.4fnn', mae);

% Plot predicted vs. actual compressive strength
figure;
scatter(y_test, y_pred);
hold on;
plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--'); % Reference line
xlabel('Actual Compressive Strength (MPa)');
ylabel('Predicted Compressive Strength (MPa)');
title('Predicted vs. Actual Compressive Strength (Linear Model)');
grid on;
saveas(gcf, 'linear_model_prediction_plot.jpg');

% Plot residuals
residuals = y_test - y_pred;
figure;
scatter(y_pred, residuals);
hold on;
plot([min(y_pred), max(y_pred)], [0, 0], 'r--'); % Reference line at zero
xlabel('Predicted Compressive Strength (MPa)');
ylabel('Residuals');
title('Residual Plot (Linear Model)');
grid on;
saveas(gcf, 'linear_model_residual_plot.jpg');

%% Step 8 Nonlinear Regression
% Define the nonlinear model
modelfun = @(b, X) b(1) + ... % Intercept
                   b(2)X(,1) + ... % Cement (linear)
                   b(3)X(,2) + ... % Blast Furnace Slag (linear)
                   b(4)X(,3) + ... % Fly Ash (linear)
                   b(5)X(,4) + ... % Water (linear)
                   b(6)X(,5) + ... % Superplasticizer (linear)
                   b(7)X(,6) + ... % Coarse Aggregate (linear)
                   b(8)X(,7) + ... % Fine Aggregate (linear)
                   b(9)X(,8) + ... % Age (linear)
                   b(10)X(,1).^2 + ... % Cement (quadratic)
                   b(11)X(,5).^2 + ... % Superplasticizer (quadratic)
                   b(12)X(,8).^2;      % Age (quadratic)

% Initial guess for coefficients
beta0 = [1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.001, 0.001, 0.001];

% Fit the nonlinear model
nonlinear_model = fitnlm(X_train, y_train, modelfun, beta0);

% Predict compressive strength on the testing data
y_pred_nl = predict(nonlinear_model, X_test);

% Calculate evaluation metrics
ss_total_nl = sum((y_test - mean(y_test)).^2);       % Total sum of squares
ss_residual_nl = sum((y_test - y_pred_nl).^2);      % Residual sum of squares
r2_score_nl = 1 - (ss_residual_nl  ss_total_nl);   % R² score
rmse_nl = sqrt(mean((y_test - y_pred_nl).^2));      % Root Mean Squared Error
mae_nl = mean(abs(y_test - y_pred_nl));             % Mean Absolute Error

% Display the evaluation metrics
disp('Nonlinear Model Performance Metrics');
fprintf('R² Score %.4fn', r2_score_nl);
fprintf('RMSE %.4fn', rmse_nl);
fprintf('MAE %.4fn', mae_nl);

% Plot predicted vs. actual compressive strength
figure;
scatter(y_test, y_pred_nl);
hold on;
plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--'); % Reference line
xlabel('Actual Compressive Strength (MPa)');
ylabel('Predicted Compressive Strength (MPa)');
title('Predicted vs. Actual Compressive Strength (Nonlinear Model)');
grid on;
saveas(gcf, 'nonlinear_model_prediction_plot.jpg');

% Plot residuals
residuals_nl = y_test - y_pred_nl;
figure;
scatter(y_pred_nl, residuals_nl);
hold on;
plot([min(y_pred_nl), max(y_pred_nl)], [0, 0], 'r--'); % Reference line at zero
xlabel('Predicted Compressive Strength (MPa)');
ylabel('Residuals');
title('Residual Plot (Nonlinear Model)');
grid on;
saveas(gcf, 'nonlinear_model_residual_plot.jpg');

%% Task 7 Error Analysis
% Predicted vs. Actual Values for Both Models
figure;
subplot(1, 2, 1);
scatter(y_test, y_pred);
hold on;
plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--'); % Reference line
xlabel('Actual Compressive Strength (MPa)');
ylabel('Predicted Compressive Strength (MPa)');
title('Linear Model Predicted vs. Actual');
grid on;

subplot(1, 2, 2);
scatter(y_test, y_pred_nl);
hold on;
plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--'); % Reference line
xlabel('Actual Compressive Strength (MPa)');
ylabel('Predicted Compressive Strength (MPa)');
title('Nonlinear Model Predicted vs. Actual');
grid on;
saveas(gcf, 'predicted_vs_actual_both_models.jpg');

% Residuals for Both Models
residuals_linear = y_test - y_pred;       % Residuals for linear model
residuals_nonlinear = y_test - y_pred_nl; % Residuals for nonlinear model

% Residual Plots for Both Models
figure;
subplot(1, 2, 1);
scatter(y_pred, residuals_linear);
hold on;
plot([min(y_pred), max(y_pred)], [0, 0], 'r--'); % Reference line at zero
xlabel('Predicted Compressive Strength (MPa)');
ylabel('Residuals');
title('Residual Plot (Linear Model)');
grid on;

subplot(1, 2, 2);
scatter(y_pred_nl, residuals_nonlinear);
hold on;
plot([min(y_pred_nl), max(y_pred_nl)], [0, 0], 'r--'); % Reference line at zero
xlabel('Predicted Compressive Strength (MPa)');
ylabel('Residuals');
title('Residual Plot (Nonlinear Model)');
grid on;
saveas(gcf, 'residual_plots_both_models.jpg');

%% Task 8 Model Performance Discussion
% Compare R² scores, RMSE, and MAE for both models
fprintf('Linear Model Performancen');
fprintf('R² Score %.4fn', r2_score_linear);
fprintf('RMSE %.4fn', rmse);
fprintf('MAE %.4fnn', mae);

fprintf('Nonlinear Model Performancen');
fprintf('R² Score %.4fn', r2_score_nl);
fprintf('RMSE %.4fn', rmse_nl);
fprintf('MAE %.4fnn', mae_nl);

% Determine which model performs better
if r2_score_nl  r2_score_linear
    fprintf('The nonlinear model performs better based on R² score.n');
else
    fprintf('The linear model performs better based on R² score.n');
end

if rmse_nl  rmse
    fprintf('The nonlinear model performs better based on RMSE.n');
else
    fprintf('The linear model performs better based on RMSE.n');
end

if mae_nl  mae
    fprintf('The nonlinear model performs better based on MAE.n');
else
    fprintf('The linear model performs better based on MAE.n');
end

%% Save the linear and nonlinear models
% Save the linear model to a .mat file
save('linear_model.mat', 'linear_model');

% Save the nonlinear model to a .mat file
save('nonlinear_model.mat', 'nonlinear_model');

% Display the coefficients of the nonlinear model
disp('Nonlinear Model Coefficients');
disp(nonlinear_model.Coefficients.Estimate);
