% This BARON input file was generated by the MATLAB/BARON interface.
% The BARON/MATLAB interface was developed by J. Currie.  The interface
% is provided free of charge and with no warranties from The Optimization
% Firm, http://www.minlp.com.
% Interface version: v1.51 [17-Dec-2013]
% File generated: 06-Jun-2016 04:10:05
% Model: model14

% Objective
fun = @(x)(2.5134-x(1)-x(3)-x(5))^2+(2.04433-(exp(-0.05*x(2))*x(1)+exp(-0.05*x(4))*x(3)+exp(-0.05*x(6))*x(5)))^2+(1.6684-(exp(-0.1*x(2))*x(1)+exp(-0.1*x(4))*x(3)+exp(-0.1*x(6))*x(5)))^2+(1.36642-(exp(-0.15*x(2))*x(1)+exp(-0.15*x(4))*x(3)+exp(-0.15*x(6))*x(5)))^2+(1.12323-(exp(-0.2*x(2))*x(1)+exp(-0.2*x(4))*x(3)+exp(-0.2*x(6))*x(5)))^2+(0.92689-(exp(-0.25*x(2))*x(1)+exp(-0.25*x(4))*x(3)+exp(-0.25*x(6))*x(5)))^2+(0.767934-(exp(-0.3*x(2))*x(1)+exp(-0.3*x(4))*x(3)+exp(-0.3*x(6))*x(5)))^2+(0.638878-(exp(-0.35*x(2))*x(1)+exp(-0.35*x(4))*x(3)+exp(-0.35*x(6))*x(5)))^2+(0.533784-(exp(-0.4*x(2))*x(1)+exp(-0.4*x(4))*x(3)+exp(-0.4*x(6))*x(5)))^2+(0.447936-(exp(-0.45*x(2))*x(1)+exp(-0.45*x(4))*x(3)+exp(-0.45*x(6))*x(5)))^2+(0.377585-(exp(-0.5*x(2))*x(1)+exp(-0.5*x(4))*x(3)+exp(-0.5*x(6))*x(5)))^2+(0.319739-(exp(-0.55*x(2))*x(1)+exp(-0.55*x(4))*x(3)+exp(-0.55*x(6))*x(5)))^2+(0.272013-(exp(-0.6*x(2))*x(1)+exp(-0.6*x(4))*x(3)+exp(-0.6*x(6))*x(5)))^2+(0.232497-(exp(-0.65*x(2))*x(1)+exp(-0.65*x(4))*x(3)+exp(-0.65*x(6))*x(5)))^2+(0.199659-(exp(-0.7*x(2))*x(1)+exp(-0.7*x(4))*x(3)+exp(-0.7*x(6))*x(5)))^2+(0.17227-(exp(-0.75*x(2))*x(1)+exp(-0.75*x(4))*x(3)+exp(-0.75*x(6))*x(5)))^2+(0.149341-(exp(-0.8*x(2))*x(1)+exp(-0.8*x(4))*x(3)+exp(-0.8*x(6))*x(5)))^2+(0.13007-(exp(-0.85*x(2))*x(1)+exp(-0.85*x(4))*x(3)+exp(-0.85*x(6))*x(5)))^2+(0.113812-(exp(-0.9*x(2))*x(1)+exp(-0.9*x(4))*x(3)+exp(-0.9*x(6))*x(5)))^2+(0.100042-(exp(-0.95*x(2))*x(1)+exp(-0.95*x(4))*x(3)+exp(-0.95*x(6))*x(5)))^2+(0.0883321-(exp(-x(2))*x(1)+exp(-x(4))*x(3)+exp(-x(6))*x(5)))^2+(0.0783354-(exp(-1.05*x(2))*x(1)+exp(-1.05*x(4))*x(3)+exp(-1.05*x(6))*x(5)))^2+(0.0697669-(exp(-1.1*x(2))*x(1)+exp(-1.1*x(4))*x(3)+exp(-1.1*x(6))*x(5)))^2+(0.0623931-(exp(-1.15*x(2))*x(1)+exp(-1.15*x(4))*x(3)+exp(-1.15*x(6))*x(5)))^2;

% Bounds
lb = [0,0,0,0,0,0]';
ub = [60,60,60,60,60,60]';

% Constraints
nlcon = [];
cl = [];
cu = [];

% Variables (C = continuous, B = binary, I = integer)
xtype = 'CCCCCC';

% Starting Guess
x0 = [NaN,NaN,NaN,NaN,NaN,NaN]';

% Options
opts = struct('probname','model14');
opts.sense = 'min';
