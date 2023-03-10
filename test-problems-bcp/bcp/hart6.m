% This BARON input file was generated by the MATLAB/BARON interface.
% The BARON/MATLAB interface was developed by J. Currie.  The interface
% is provided free of charge and with no warranties from The Optimization
% Firm, http://www.minlp.com.
% Interface version: v1.51 [17-Dec-2013]
% File generated: 06-Jun-2016 04:06:02
% Model: hart6

% Objective
fun = @(x)-(exp(-(10*(-0.1312+x(1))^2+0.05*(-0.1696+x(2))^2+17*(-0.5569+x(3))^2+3.5*(-0.0124+x(4))^2+1.7*(-0.8283+x(5))^2+8*(-0.5886+x(6))^2))+1.2*exp(-(0.05*(-0.2329+x(1))^2+10*(-0.4135+x(2))^2+17*(-0.8307+x(3))^2+0.1*(-0.3736+x(4))^2+8*(-0.1004+x(5))^2+14*(-0.9991+x(6))^2))+3*exp(-(3*(-0.2348+x(1))^2+3.5*(-0.1451+x(2))^2+1.7*(-0.3522+x(3))^2+10*(-0.2883+x(4))^2+17*(-0.3047+x(5))^2+8*(-0.665+x(6))^2))+3.2*exp(-(17*(-0.4047+x(1))^2+8*(-0.8828+x(2))^2+0.05*(-0.8732+x(3))^2+10*(-0.5743+x(4))^2+0.1*(-0.1091+x(5))^2+14*(-0.0381+x(6))^2)));

% Bounds
lb = [0,0,0,0,0,0]';
ub = [1,1,1,1,1,1]';

% Constraints
nlcon = [];
cl = [];
cu = [];

% Variables (C = continuous, B = binary, I = integer)
xtype = 'CCCCCC';

% Starting Guess
x0 = [0.2,0.2,0.2,0.2,0.2,0.2]';

% Options
opts = struct('probname','hart6');
opts.sense = 'min';
