% This BARON input file was generated by the MATLAB/BARON interface.
% The BARON/MATLAB interface was developed by J. Currie.  The interface
% is provided free of charge and with no warranties from The Optimization
% Firm, http://www.minlp.com.
% Interface version: v1.51 [17-Dec-2013]
% File generated: 06-Jun-2016 04:10:05
% Model: model39

% Objective
fun = @(x)(4-(exp(2*x(1))+exp(2*x(2))))^2+(4.1-(exp(2.1*x(1))+exp(2.1*x(2))))^2+(4.3-(exp(2.2*x(1))+exp(2.2*x(2))))^2+(4.5-(exp(2.3*x(1))+exp(2.3*x(2))))^2+(4.6-(exp(2.4*x(1))+exp(2.4*x(2))))^2+(4.8-(exp(2.5*x(1))+exp(2.5*x(2))))^2+(5-(exp(2.6*x(1))+exp(2.6*x(2))))^2+(5.1-(exp(2.7*x(1))+exp(2.7*x(2))))^2+(5.3-(exp(2.8*x(1))+exp(2.8*x(2))))^2+(5.5-(exp(2.9*x(1))+exp(2.9*x(2))))^2;

% Bounds
lb = [-50,-50]';
ub = [50,50]';

% Constraints
nlcon = [];
cl = [];
cu = [];

% Variables (C = continuous, B = binary, I = integer)
xtype = 'CC';

% Starting Guess
x0 = [NaN,NaN]';

% Options
opts = struct('probname','model39');
opts.sense = 'min';
