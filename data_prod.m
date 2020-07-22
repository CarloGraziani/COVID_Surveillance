clear all;
N_days = 30.0;
days = 0.0:1.0:N_days;
data = zeros(3, N_days);
vpar = [];
init_state_global = [];
a = 30;
b = 500;
for i = 1:N_days+1
    N_xt = (b-a).*rand(1) + a;
    C_xt = rand(1) * N_xt;
    data_today = [days(i); cast(ceil(N_xt),'double')  ; cast(ceil(C_xt), 'double')];
    data(:,i) = data_today;
%     %%%%%%%vpar%%%%%%%%%
%     index = 1;
%     mu_b = 5; sigma_b = 1;
%     beta = mu_b + 1. * randn(1);
%     L = 0.0025/beta;
%     par = [L,0.01,beta*1E-7,0.5,20.0,10.0];
%     V0 = 1e3 + 1e2 * randn(1);
%     X0 = 1e6;
%     Y0 = V0;
%     init_state = [V0, X0, Y0];
%     while index <= N_xt - 1
%         mu_b = 5; sigma_b = 1;
%         beta = mu_b + 1. * randn(1);
%         L = 0.0025/beta;
%         par_new = [L,0.01,beta*1E-7,0.5,20.0,10.0];
%         par = [par; par_new];
%         V0 = 1e3 + 1e2 * randn(1);
%         X0 = 1e6;
%         Y0 = V0;
%         init_state_new = [V0, X0, Y0];
%         init_state = [init_state; init_state_new];
%         index = index + 1;
%     end
%     i
%     vpar(i,:,:) = par;
%     init_state_global(i,:,:) = init_state;
%         
        
end


save('data/test_data','data')

%save('data/vpar','vpar','init_state_global')