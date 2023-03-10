% This BARON input file was generated by the MATLAB/BARON interface.
% The BARON/MATLAB interface was developed by J. Currie.  The interface
% is provided free of charge and with no warranties from The Optimization
% Firm, http://www.minlp.com.
% Interface version: v1.51 [17-Dec-2013]
% File generated: 06-Jun-2016 04:05:49
% Model: draperj

% Objective
fun = @(x)(0.126-x(1)*x(3)/(1+x(1)+x(2)))^2+(0.219-2*x(1)*x(3)/(1+2*x(1)+x(2)))^2+(0.076-x(1)*x(3)/(1+x(1)+2*x(2)))^2+(0.126-2*x(1)*x(3)/(1+2*x(1)+2*x(2)))^2+(0.186-0.1*x(1)*x(3)/(1+0.1*x(1)))^2+(0.606-3*x(1)*x(3)/(1+3*x(1)))^2+(0.268-0.2*x(1)*x(3)/(1+0.2*x(1)))^2+(0.614-3*x(1)*x(3)/(1+3*x(1)))^2+(0.318-0.3*x(1)*x(3)/(1+0.3*x(1)))^2+(0.298-3*x(1)*x(3)/(1+3*x(1)+0.8*x(2)))^2+(0.509-3*x(1)*x(3)/(1+3*x(1)))^2+(0.247-0.2*x(1)*x(3)/(1+0.2*x(1)))^2+(0.319-3*x(1)*x(3)/(1+3*x(1)+0.8*x(2)))^2;

% Bounds
lb = [0,0,0]';
ub = [1000,1000,1000]';

% Constraints
nlcon = [];
cl = [];
cu = [];

% Variables (C = continuous, B = binary, I = integer)
xtype = 'CCC';

% Starting Guess
x0 = [NaN,NaN,NaN]';

% Options
opts = struct('probname','draperj');
opts.sense = 'min';
