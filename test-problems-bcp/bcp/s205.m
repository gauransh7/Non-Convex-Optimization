% This BARON input file was generated by the MATLAB/BARON interface.
% The BARON/MATLAB interface was developed by J. Currie.  The interface
% is provided free of charge and with no warranties from The Optimization
% Firm, http://www.minlp.com.
% Interface version: v1.51 [17-Dec-2013]
% File generated: 06-Jun-2016 04:12:09
% Model: s205

% Objective
fun = @(x)(1.5-x(1)*(1-x(2)))^2+(2.25-(1-x(2)^2)*x(1))^2+(2.625-(1-x(2)^3)*x(1))^2;

% Bounds
lb = [-Inf,-Inf]';
ub = [Inf,Inf]';

% Constraints
nlcon = [];
cl = [];
cu = [];

% Variables (C = continuous, B = binary, I = integer)
xtype = 'CC';

% Starting Guess
x0 = [NaN,NaN]';

% Options
opts = struct('probname','s205');
opts.sense = 'min';
