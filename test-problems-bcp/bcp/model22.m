% This BARON input file was generated by the MATLAB/BARON interface.
% The BARON/MATLAB interface was developed by J. Currie.  The interface
% is provided free of charge and with no warranties from The Optimization
% Firm, http://www.minlp.com.
% Interface version: v1.51 [17-Dec-2013]
% File generated: 06-Jun-2016 04:10:05
% Model: model22

% Objective
fun = @(x)(10.07-77.6*x(1)*x(2)*(1+77.6*x(2))^(-1))^2+(14.73-114.9*x(1)*x(2)*(1+114.9*x(2))^(-1))^2+(17.94-141.1*x(1)*x(2)*(1+141.1*x(2))^(-1))^2+(23.93-190.8*x(1)*x(2)*(1+190.8*x(2))^(-1))^2+(29.61-239.9*x(1)*x(2)*(1+239.9*x(2))^(-1))^2+(35.18-289*x(1)*x(2)*(1+289*x(2))^(-1))^2+(40.02-332.8*x(1)*x(2)*(1+332.8*x(2))^(-1))^2+(44.82-378.4*x(1)*x(2)*(1+378.4*x(2))^(-1))^2+(50.76-434.8*x(1)*x(2)*(1+434.8*x(2))^(-1))^2+(55.05-477.3*x(1)*x(2)*(1+477.3*x(2))^(-1))^2+(61.01-536.8*x(1)*x(2)*(1+536.8*x(2))^(-1))^2+(66.4-593.1*x(1)*x(2)*(1+593.1*x(2))^(-1))^2+(75.47-689.1*x(1)*x(2)*(1+689.1*x(2))^(-1))^2+(81.78-760*x(1)*x(2)*(1+760*x(2))^(-1))^2;

% Bounds
lb = [0,0]';
ub = [1000,1000]';

% Constraints
nlcon = [];
cl = [];
cu = [];

% Variables (C = continuous, B = binary, I = integer)
xtype = 'CC';

% Starting Guess
x0 = [NaN,NaN]';

% Options
opts = struct('probname','model22');
opts.sense = 'min';
