%% Load data
load data/fAttasElv1.mat

t = fAttasElv1(:, 1);
de = fAttasElv1(:, 22);
q = fAttasElv1(:, 8);

equilibrium = t < 5.24;
de_eq = mean(de(equilibrium));
q_bias = mean(q(equilibrium));

Ts = diff(t(1:2));

data_est = iddata(q - q_bias, de - de_eq, Ts);

%% Run discrete-time output-error estimation
opt = oeOptions;
opt.Display = 'on';

%sys0 = idpoly(1, [1, 0],[], [], [1,0,0], 1, Ts);
sys = oe(data_est, [2, 2, 0], opt);
