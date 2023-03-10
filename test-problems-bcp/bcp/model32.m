% This BARON input file was generated by the MATLAB/BARON interface.
% The BARON/MATLAB interface was developed by J. Currie.  The interface
% is provided free of charge and with no warranties from The Optimization
% Firm, http://www.minlp.com.
% Interface version: v1.51 [17-Dec-2013]
% File generated: 06-Jun-2016 04:10:05
% Model: model32

% Objective
fun = @(x)(40.2-2*x(3))^2+(11.0349-(exp(-0.6*x(1))+exp(-0.4*x(2)))*x(3))^2+(4.48869-(exp(-0.6*x(1))+exp(-x(2)))*x(3))^2+(2.46137-(exp(-1.4*x(1))+exp(-1.4*x(2)))*x(3))^2+(2.45137-(exp(-2.6*x(1))+exp(-1.4*x(2)))*x(3))^2+(1.82343-(exp(-3.2*x(1))+exp(-1.6*x(2)))*x(3))^2+(1.00094-(exp(-0.8*x(1))+exp(-2*x(2)))*x(3))^2+(0.741352-(exp(-1.6*x(1))+exp(-2.2*x(2)))*x(3))^2+(0.741352-(exp(-2.6*x(1))+exp(-2.2*x(2)))*x(3))^2+(0.741352-(exp(-4*x(1))+exp(-2.2*x(2)))*x(3))^2+(0.406863-(exp(-1.2*x(1))+exp(-2.6*x(2)))*x(3))^2+(0.406862-(exp(-2*x(1))+exp(-2.6*x(2)))*x(3))^2+(0.301411-(exp(-4.6*x(1))+exp(-2.8*x(2)))*x(3))^2+(0.223291-(exp(-3.2*x(1))+exp(-3*x(2)))*x(3))^2+(0.165418-(exp(-1.6*x(1))+exp(-3.2*x(2)))*x(3))^2+(0.122545-(exp(-4.2*x(1))+exp(-3.4*x(2)))*x(3))^2+(0.067254-(exp(-2*x(1))+exp(-3.8*x(2)))*x(3))^2+(0.067254-(exp(-3.2*x(1))+exp(-3.8*x(2)))*x(3))^2+(0.03691-(exp(-2.8*x(1))+exp(-4.2*x(2)))*x(3))^2+(0.03691-(exp(-4.2*x(1))+exp(-4.2*x(2)))*x(3))^2+(0.027343-(exp(-5.4*x(1))+exp(-4.4*x(2)))*x(3))^2+(0.015006-(exp(-5.6*x(1))+exp(-4.8*x(2)))*x(3))^2+(0.011117-(exp(-3.2*x(1))+exp(-5*x(2)))*x(3))^2;

% Bounds
lb = [0,0,0]';
ub = [1000,1000,10000]';

% Constraints
nlcon = [];
cl = [];
cu = [];

% Variables (C = continuous, B = binary, I = integer)
xtype = 'CCC';

% Starting Guess
x0 = [NaN,NaN,NaN]';

% Options
opts = struct('probname','model32');
opts.sense = 'min';
