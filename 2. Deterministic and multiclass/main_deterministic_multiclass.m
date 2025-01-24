% % Title of the project:
% % A Novel Robust Optimization Model for Nonlinear Support Vector Machine

% % Authors:
% % Francesca Maggioni and Andrea Spinelli (University of Bergamo, IT)

% % Reference:
% % F. Maggioni, A. Spinelli. (2025). _A Novel Robust Optimization Model for Nonlinear Support Vector Machine_. European Journal of Operational Research, 322(1), 237-253. https://doi.org/10.1016/j.ejor.2024.12.014


% % Case: deterministic and multiclass classifier
format long

clear
close all
clc

DATA = readtable('iris_multiclass.csv');

n_runs = 96;
testing_error = zeros(n_runs,1);

tic
parfor i_runs = 1:n_runs
       [testing_error(i_runs)]...
           = unit_of_work_deterministic_multiclass(DATA);
end
toc

disp('mean testing error')
mean_all = mean(testing_error)

disp('std testing error')
std_all = std(testing_error)


