% This BARON input file was generated by the MATLAB/BARON interface.
% The BARON/MATLAB interface was developed by J. Currie.  The interface
% is provided free of charge and with no warranties from The Optimization
% Firm, http://www.minlp.com.
% Interface version: v1.51 [17-Dec-2013]
% File generated: 06-Jun-2016 04:12:29
% Model: stattools

% Objective
fun = @(x)(0.1913-(1-exp(-0.1*x(2)))*x(1))^2+(0.0737-(1-exp(-0.2*x(2)))*x(1))^2+(0.2702-(1-exp(-0.3*x(2)))*x(1))^2+(0.427-(1-exp(-0.4*x(2)))*x(1))^2+(0.2968-(1-exp(-0.5*x(2)))*x(1))^2+(0.4474-(1-exp(-0.6*x(2)))*x(1))^2+(0.4941-(1-exp(-0.7*x(2)))*x(1))^2+(0.5682-(1-exp(-0.8*x(2)))*x(1))^2+(0.563-(1-exp(-0.9*x(2)))*x(1))^2+(0.6636-(1-exp(-x(2)))*x(1))^2;

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
opts = struct('probname','stattools');
opts.sense = 'min';
