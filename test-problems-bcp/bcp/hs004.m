% This BARON input file was generated by the MATLAB/BARON interface.
% The BARON/MATLAB interface was developed by J. Currie.  The interface
% is provided free of charge and with no warranties from The Optimization
% Firm, http://www.minlp.com.
% Interface version: v1.51 [17-Dec-2013]
% File generated: 06-Jun-2016 04:06:03
% Model: hs004

% Objective
fun = @(x)0.333333333333333*(1+x(1))^3+x(2);

% Bounds
lb = [1,0]';
ub = [Inf,Inf]';

% Constraints
nlcon = [];
cl = [];
cu = [];

% Variables (C = continuous, B = binary, I = integer)
xtype = 'CC';

% Starting Guess
x0 = [1.125,0.125]';

% Options
opts = struct('probname','hs004');
opts.sense = 'min';
