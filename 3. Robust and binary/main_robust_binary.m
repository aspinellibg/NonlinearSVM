% % Title of the project:
% % A Novel Robust Optimization Model for Nonlinear Support Vector Machine

% % Authors:
% % Francesca Maggioni and Andrea Spinelli (University of Bergamo, IT)

% % Reference:
% % Preprint available at https://arxiv.org/abs/2306.06223


% % Case: robust and binary classifier
format long

clear
close all
clc

DATA = readtable('mammographicmass_binary.csv');

n_runs = 96;
vect_rho = logspace(-7,-1,7);
num_rho = length(vect_rho);
testing_error = zeros(n_runs,num_rho);
testing_error_A = zeros(n_runs,num_rho);
testing_error_B = zeros(n_runs,num_rho);
tic
for index_rho = 1:num_rho
    scalar_rho = vect_rho(index_rho);
    parfor i_runs = 1:n_runs
        [testing_error(i_runs,index_rho),...
            testing_error_A(i_runs,index_rho),...
            testing_error_B(i_runs,index_rho)] = unit_of_work_robust_binary(DATA,scalar_rho);
    end
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

