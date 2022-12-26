% This BARON input file was generated by the MATLAB/BARON interface.
% The BARON/MATLAB interface was developed by J. Currie.  The interface
% is provided free of charge and with no warranties from The Optimization
% Firm, http://www.minlp.com.
% Interface version: v1.51 [17-Dec-2013]
% File generated: 06-Jun-2016 04:05:58
% Model: ex8_1_6

% Objective
fun = @(x)(-1/(0.1+(-4+x(1))^2+(-4+x(2))^2))-1/(0.2+(-1+x(1))^2+(-1+x(2))^2)-1/(0.2+(-8+x(1))^2+(-8+x(2))^2);

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
opts = struct('probname','ex8_1_6');
opts.sense = 'min';
