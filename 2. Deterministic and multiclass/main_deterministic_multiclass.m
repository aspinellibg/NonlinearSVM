% % Title of the project:
% % A Novel Robust Optimization Model for Nonlinear Support Vector Machine

% % Authors:
% % Francesca Maggioni and Andrea Spinelli (University of Bergamo, IT)

% % Reference:
% % Preprint available at https://arxiv.org/abs/2306.06223


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


