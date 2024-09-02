**Description**

The codes provided in this folder are related to the project "_A Novel Robust Optimization Model for Nonlinear Support Vector Machine_".

All the codes are written in MATLAB. The models are solved using CVX and MOSEK. Please visit https://cvxr.com/cvx/ and https://www.mosek.com for details and licensing issues.

We provide four different folders, depending on the choice of the classifier (deterministic vs robust; binary vs multiclass).
Specifically, each folder contains the following files:
  - an example of dataset in .csv ("mammographicmass" for the binary classifier and "iris" for the multiclass classifier, for details please visit https://archive.ics.uci.edu). The last column of the table refers to the labels of the observations (1-0 binary; 1-2-3-... multiclass);
  - a main code in .m: the code that has to be run in MATLAB;
  - a unit of work code in .m: the user has to properly choose the kernel function and the relevant parameters of the implementation;
  - a holdouts training-testing file in .m: a code to split the dataset into training set and testing set, according to a paramter set by the user in the unit of work code.

For all the details of the implementation, the user is referred to the reference reported as follows.

**Reference**

F. Maggioni, A. Spinelli. (2024). _A Novel Robust Optimization Model for Nonlinear Support Vector Machine_. Submitted, under second revision. Preprint available at https://arxiv.org/abs/2306.06223
