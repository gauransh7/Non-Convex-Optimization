% This BARON input file was generated by the MATLAB/BARON interface.
% The BARON/MATLAB interface was developed by J. Currie.  The interface
% is provided free of charge and with no warranties from The Optimization
% Firm, http://www.minlp.com.
% Interface version: v1.51 [17-Dec-2013]
% File generated: 06-Jun-2016 04:12:10
% Model: s296

% Objective
fun = @(x)100*(x(2)-x(1)^2)^2+(1-x(1))^2+100*(x(3)-x(2)^2)^2+(1-x(2))^2+100*(x(4)-x(3)^2)^2+(1-x(3))^2+100*(x(5)-x(4)^2)^2+(1-x(4))^2+100*(x(6)-x(5)^2)^2+(1-x(5))^2+100*(x(7)-x(6)^2)^2+(1-x(6))^2+100*(x(8)-x(7)^2)^2+(1-x(7))^2+100*(x(9)-x(8)^2)^2+(1-x(8))^2+100*(x(10)-x(9)^2)^2+(1-x(9))^2+100*(x(11)-x(10)^2)^2+(1-x(10))^2+100*(x(12)-x(11)^2)^2+(1-x(11))^2+100*(x(13)-x(12)^2)^2+(1-x(12))^2+100*(x(14)-x(13)^2)^2+(1-x(13))^2+100*(x(15)-x(14)^2)^2+(1-x(14))^2+100*(x(16)-x(15)^2)^2+(1-x(15))^2;

% Bounds
lb = [-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf]';
ub = [Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf]';

% Constraints
nlcon = [];
cl = [];
cu = [];

% Variables (C = continuous, B = binary, I = integer)
xtype = 'CCCCCCCCCCCCCCCC';

% Starting Guess
x0 = [-1.2,1,-1.2,1,-1.2,1,-1.2,1,-1.2,1,-1.2,1,-1.2,1,-1.2,1]';

% Options
opts = struct('probname','s296');
opts.sense = 'min';
