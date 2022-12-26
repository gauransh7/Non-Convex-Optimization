% This BARON input file was generated by the MATLAB/BARON interface.
% The BARON/MATLAB interface was developed by J. Currie.  The interface
% is provided free of charge and with no warranties from The Optimization
% Firm, http://www.minlp.com.
% Interface version: v1.51 [17-Dec-2013]
% File generated: 06-Jun-2016 04:11:50
% Model: pfit2ls

% Objective
fun = @(x)(26.6666666666667+x(1)*x(2)*x(3)-0.5*x(1)*(1+x(1))*x(2)*x(3)*x(3)-(1-exp(-log(1+x(3))*x(1)))*x(2))^2+(60.4444444444444+x(1)*x(2)*x(3)*(1-exp(log(1+x(3))*(-1-x(1))))-x(1)*(1+x(1))*x(2)*x(3)*x(3))^2+(71.1111111111111-x(1)*(1+x(1))*x(2)*x(3)*x(3)*(1-exp(log(1+x(3))*(-2-x(1)))))^2;

% Bounds
lb = [-Inf,-Inf,-Inf]';
ub = [Inf,Inf,Inf]';

% Constraints
nlcon = [];
cl = [];
cu = [];

% Variables (C = continuous, B = binary, I = integer)
xtype = 'CCC';

% Starting Guess
x0 = [1,NaN,1]';

% Options
opts = struct('probname','pfit2ls');
opts.sense = 'min';
