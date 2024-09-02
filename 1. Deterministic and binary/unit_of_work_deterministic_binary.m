function [testing_error, testing_error_A, testing_error_B] = unit_of_work_deterministic_binary(DATA)

% % training vs testing sets
testingsamplesize = 0.25;
[Atrain, Atest, Btrain, Btest] = holdouts_train_test(DATA, testingsamplesize);

dati_set_A = Atrain';
dati_set_B = Btrain';

[~,m_A] = size(dati_set_A);
[~,m_B] = size(dati_set_B);

dati = [dati_set_A dati_set_B];
[n,m] = size(dati);

y = [ones(1,m_A) -ones(1,m_B)];
D = diag(y);

% % choice of the kernel function
% % NB:
% % change accordingly in the testing phase too (lines 121-122 and 135-136)
K = zeros(m,m);

% % polynomial kernel --> k(x,y) = (<x,y>+c)^d
d = 2; c = max(std(dati,0,2));
i = 1; j = 1;
while (i<=m)
    while (j<=m)
        K(i,j) = (dati(:,i)'*dati(:,j)+c)^d;
        j = j+1;
    end
    i = i+1;
    j = i;
end
K = K+K';
for i = 1:m
    K(i,i) = K(i,i)/2;
end

% % Gaussian RBF kernel --> k(x,y) = exp(-norm(x-y)^2/(2*alpha^2))
% alpha = max(std(dati,0,2));
% i = 1; j = 1;
% while (i<=m)
%     while (j<=m)
%         K(i,j) = exp(-norm(dati(:,i)-dati(:,j))^2/(2*alpha^2));
%         j = j+1;
%     end
%     i = i+1;
%     j = i;
% end
% K = K+K';
% for i = 1:m
%     K(i,i) = K(i,i)/2;
% end

% % TRAINING PHASE
vectornu = logspace(-3,0,5);

training_error_opt = Inf;

for i_nu = 1:length(vectornu)
    nu = vectornu(i_nu);

    cvx_begin quiet
    cvx_solver mosek
    cvx_precision high
    variables u(m) vargamma xi(m) s(m)
    minimize sum(s) + nu*sum(xi)
    subject to
    D*(K*D*u-ones(m,1)*vargamma)+xi >= ones(m,1);
    xi >= 0;
    u >= -s;
    u <= s;
    s >= 0;
    cvx_end;

    omega_A = max(D*xi);
    omega_B = max(-D*xi);

    num_points = 1e4;
    discr_b = linspace(vargamma+1-omega_B,vargamma-1+omega_A,num_points);

    b_opt = vargamma;
    max_b = m;
    for j = 1:length(discr_b)
        b = discr_b(j);
        if sum((D*(-K*D*u+ones(m,1)*b))>0) < max_b
            max_b = sum(D*(-K*D*u+ones(m,1)*b)>0);
            b_opt = b;
        end
    end

    tot_num_misclass_training = length(find(D*(-K*D*u+ones(m,1)*b_opt)>0));
    training_error = tot_num_misclass_training/m;

    if training_error < training_error_opt
        training_error_opt = training_error;
        u_opt = u;
        b_opt_opt = b_opt;
    end

end

% % TESTING PHASE
Atest = Atest';
Btest = Btest';

[~,m_Atest] = size(Atest);
[~,m_Btest] = size(Btest);

truepositive = 0;
falsepositive = 0;
truenegative = 0;
falsenegative = 0;

for i = 1:m_Atest
    K_test = zeros(m,1);
    xtest = Atest(:,i);
    for j = 1:m
        K_test(j) = (dati(:,j)'*xtest+c)^d;
        % K_test(j) = exp(-norm(dati(:,j)-xtest)^2/(2*alpha^2));
    end
    if (K_test'*D*u_opt-b_opt_opt > 0)
        truepositive = truepositive + 1;
    else
        falsenegative = falsenegative + 1;
    end
end

for i = 1:m_Btest
    K_test = zeros(m,1);
    xtest = Btest(:,i);
    for j = 1:m
        K_test(j) = (dati(:,j)'*xtest+c)^d;
        %K_test(j) = exp(-norm(dati(:,j)-xtest)^2/(2*alpha^2));
    end
    if (K_test'*D*u_opt-b_opt_opt <= 0)
        truenegative = truenegative + 1;
    else
        falsepositive = falsepositive + 1;
    end
end

tot_num_misclass_testing = falsenegative + falsepositive;
testing_error = tot_num_misclass_testing/(m_Atest+m_Btest);

testing_error_A = falsenegative/m_Atest;
testing_error_B = falsepositive/m_Btest;

end