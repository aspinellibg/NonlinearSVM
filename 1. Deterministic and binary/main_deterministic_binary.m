% % Title of the project:
% % A Novel Robust Optimization Model for Nonlinear Support Vector Machine

% % Authors:
% % Francesca Maggioni and Andrea Spinelli (University of Bergamo, IT)

% % Reference:
% % Preprint available at https://arxiv.org/abs/2306.06223


% % Case: deterministic and binary classifier
format long

clear
close all
clc

DATA = readtable('mammographicmass_binary.csv');

n_runs = 96;
testing_error = zeros(n_runs,1);
testing_error_classA = zeros(n_runs,1);
testing_error_classB = zeros(n_runs,1);

tic
parfor i_runs = 1:n_runs
    [testing_error(i_runs),...
        testing_error_classA(i_runs),...
        testing_error_classB(i_runs)] = unit_of_work_deterministic_binary(DATA);
end
toc

disp('mean testing error')
mean_all = mean(testing_error)
disp('std testing error')
std_all = std(testing_error)


disp('mean testing error class A')
mean_classA = mean(testing_error_classA)
disp('std testing error class A')
std_classA = std(vect_testing_error_A)


disp('mean testing error class B')
mean_classB = mean(testing_error_classB)
disp('std testing error class B')
std_classB = std(vect_testing_error_B)


