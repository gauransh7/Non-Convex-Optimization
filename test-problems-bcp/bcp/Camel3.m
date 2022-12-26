% This BARON input file was generated by the MATLAB/BARON interface.
% The BARON/MATLAB interface was developed by J. Currie.  The interface
% is provided free of charge and with no warranties from The Optimization
% Firm, http://www.minlp.com.
% Interface version: v1.51 [17-Dec-2013]
% File generated: 06-Jun-2016 04:02:31
% Model: Camel3

% Objective
fun = @(x)(2+0.166666666666667*x(1)^4-1.05*x(1)*x(1))*x(1)*x(1)+x(1)*x(2)+x(2)*x(2);

% Bounds
lb = [-5,-5]';
ub = [5,5]';

% Constraints
nlcon = [];
cl = [];
cu = [];

% Variables (C = continuous, B = binary, I = integer)
xtype = 'CC';

% Starting Guess
x0 = [NaN,NaN]';

% Options
opts = struct('probname','Camel3');
opts.sense = 'min';