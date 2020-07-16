clear all;
N_days = 30.0;
days = 0.0:1.0:N_days;
data = [];
a = 30;
b = 500;
for i = 1:N_days+1
    N_xt = (b-a).*rand(1) + a;
    C_xt = rand(1) * N_xt;
    data_today = [days(i); cast(ceil(N_xt),'double')  ; cast(ceil(C_xt), 'double')];
    data = [data, data_today];
end

save('data/test_data','data')
