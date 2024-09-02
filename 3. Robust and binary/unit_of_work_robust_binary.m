function [testing_error, testing_error_A, testing_error_B] = unit_of_work_robust_binary(DATA,scalar_rho)

% % training vs testing sets
testingsamplesize = 0.25;
[Atrain, Atest, Btrain, Btest] = holdouts_train_test(DATA, testingsamplesize);

dati_set_A = Atrain';
dati_set_B = Btrain';

[~,m_A] = size(dati_set_A);
[~,m_B] = size(dati_set_B);

dati = [dati_set_A dati_set_B];
[n,m] = size(dati);

dati_set_A = dati(:,1:m_A);
dati_set_B = dati(:,m_A+1:end);
y = [ones(1,m_A) -ones(1,m_B)];
D = diag(y);


% % choice of the kernel function
% % NB:
% % change accordingly in lines 81-125, 191-192 and 205-206
K = zeros(m,m);

% % polynomial kernel --> k(x,y) = (<x,y>+c)^d
d = 2; c = 0;
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


% % PERTURBATION PHASE (INPUT SPACE)
% % choice of the p-norm
p = inf;

std_A = std(dati_set_A,0,2);
std_B = std(dati_set_B,0,2);
rho_A = scalar_rho;
rho_B = rho_A;
eta_A = rho_A*max(std_A);
eta_B = rho_B*max(std_B);


% % PERTURBATION PHASE (FEATURE SPACE)
if p == inf
    C = sqrt(n);
elseif p <= 2
    C = 1;
elseif p > 2
    C = n^((p-2)/(2*p));
end

% % polynomial kernel
delta_A = zeros(m_A,1);
delta_B = zeros(m_B,1);
if d > 1
    aux_j = 0;
    aux_k = 0;
    for i = 1:m_A
        x_i = dati_set_A(:,i);
        for k = 1:d
            delta_A(i) = delta_A(i) + nchoosek(d,k)*norm(x_i,2)^(d-k)*(C*eta_A)^k;
        end
        for k = 1:d-1
            for j = 1:d-k
                aux_j = aux_j + nchoosek(d-k,j)*norm(x_i,2)^(d-k-j)*(C*eta_A)^j;
            end
            aux_k = aux_k + nchoosek(d,k)*c^k*(aux_j)^2;
        end
        delta_A(i) = (delta_A(i))^2 + (aux_k)^2;
    end
    delta_A = sqrt(delta_A);
    aux_j = 0;
    aux_k = 0;
    for i = 1:m_B
        x_i = dati_set_B(:,i);
        for k = 1:d
            delta_B(i) = delta_B(i) + nchoosek(d,k)*norm(x_i,2)^(d-k)*(C*eta_B)^k;
        end
        for k = 1:d-1
            for j = 1:d-k
                aux_j = aux_j + nchoosek(d-k,j)*norm(x_i,2)^(d-k-j)*(C*eta_B)^j;
            end
            aux_k = aux_k + nchoosek(d,k)*c^k*(aux_j)^2;
        end
        delta_B(i) = (delta_B(i))^2 + (aux_k)^2;
    end
    delta_B = sqrt(delta_B);
    delta = [delta_A' delta_B'];
    delta = delta';
end

% % Gaussian RBF kernel
% delta_A = sqrt(2-2*exp(-(C*eta_A)^2/(2*alpha^2)));
% delta_B = sqrt(2-2*exp(-(C*eta_B)^2/(2*alpha^2)));
% delta = [delta_A*ones(1,m_A) delta_B*ones(1,m_B)];
% delta = delta';

% % TRAINING PHASE
K_d_sqrt = sqrt(diag(K));
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
    D*(K*D*u-ones(m,1)*vargamma)+xi-delta*(K_d_sqrt'*s) >= ones(m,1);
    xi >= 0;
    u >= -s;
    u <= s;
    s >= 0;
    cvx_end;

    omega_A = max(D*xi);
    omega_B = max(-D*xi);

    num_points = 1e4;
    discr_b = linspace(vargamma+1-omega_B,vargamma-1+omega_A,num_points);

    max_b = m;
    b_opt = vargamma;
    for j = 1:length(discr_b)
        b = discr_b(j);
        if sum((D*(-K*D*u+ones(m,1)*b)+delta*(K_d_sqrt'*abs(u)))>0) < max_b
            max_b = sum((D*(-K*D*u+ones(m,1)*b)+delta*(K_d_sqrt'*abs(u)))>0);
            b_opt = b;
        end
    end

    tot_num_misclass_training = length(find((D*(-K*D*u+ones(m,1)*b_opt)+delta*(K_d_sqrt'*abs(u)))>0));
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
        %K_test(j) = exp(-norm(dati(:,j)-xtest)^2/(2*alpha^2));
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