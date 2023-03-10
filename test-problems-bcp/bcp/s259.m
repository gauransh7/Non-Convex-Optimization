% This BARON input file was generated by the MATLAB/BARON interface.
% The BARON/MATLAB interface was developed by J. Currie.  The interface
% is provided free of charge and with no warranties from The Optimization
% Firm, http://www.minlp.com.
% Interface version: v1.51 [17-Dec-2013]
% File generated: 06-Jun-2016 04:12:09
% Model: s259

% Objective
fun = @(x)100*(x(2)-x(1)^2)^2+(1-x(1))^2+90*(x(4)-x(3)^2)^2+(1-x(3))^3+10.1*(-1+x(2))^2+(-1+x(4))^2+(-19.8+19.8*x(2))*(-1+x(4));

% Bounds
lb = [-Inf,-Inf,-Inf,-Inf]';
ub = [Inf,Inf,Inf,1]';

% Constraints
nlcon = [];
cl = [];
cu = [];

% Variables (C = continuous, B = binary, I = integer)
xtype = 'CCCC';

% Starting Guess
x0 = [NaN,NaN,NaN,NaN]';

% Options
opts = struct('probname','s259');
opts.sense = 'min';
