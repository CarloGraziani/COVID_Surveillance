clear all;
filename = 'simulation_output_100.txt';
[A,delimiterOut]=importdata(filename)
M = A.data;
[nn,mm] = size(M)
N_days = 0:1:(nn-2);
N_days = N_days';
N_xt = M(2:end,1);
C_xt = M(2:end,2);
data_simulation = [N_days, N_xt, C_xt];
data_simulation = data_simulation';
save('data/test_data_simulation','data_simulation')

pos_proba = C_xt./N_xt