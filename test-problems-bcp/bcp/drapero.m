% This BARON input file was generated by the MATLAB/BARON interface.
% The BARON/MATLAB interface was developed by J. Currie.  The interface
% is provided free of charge and with no warranties from The Optimization
% Firm, http://www.minlp.com.
% Interface version: v1.51 [17-Dec-2013]
% File generated: 06-Jun-2016 04:05:49
% Model: drapero

% Objective
fun = @(x)(-1.1-0^x(2)-x(1))^2+(-1^x(2)-x(1))^2+(2.9-2^x(2)-x(1))^2+(8.1-3^x(2)-x(1))^2;

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
opts = struct('probname','drapero');
opts.sense = 'min';
