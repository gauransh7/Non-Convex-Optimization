% This BARON input file was generated by the MATLAB/BARON interface.
% The BARON/MATLAB interface was developed by J. Currie.  The interface
% is provided free of charge and with no warranties from The Optimization
% Firm, http://www.minlp.com.
% Interface version: v1.51 [17-Dec-2013]
% File generated: 06-Jun-2016 04:11:51
% Model: pspdoc

% Objective
fun = @(x)(1+x(1)^2+(x(2)-x(3))^2)^0.5+(1+x(2)^2+(x(3)-x(4))^2)^0.5;

% Bounds
lb = [-Inf,-Inf,-Inf,-Inf]';
ub = [-1,Inf,Inf,Inf]';

% Constraints
nlcon = [];
cl = [];
cu = [];

% Variables (C = continuous, B = binary, I = integer)
xtype = 'CCCC';

% Starting Guess
x0 = [-1,3,3,3]';

% Options
opts = struct('probname','pspdoc');
opts.sense = 'min';
