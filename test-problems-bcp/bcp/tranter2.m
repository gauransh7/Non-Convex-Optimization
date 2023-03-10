% This BARON input file was generated by the MATLAB/BARON interface.
% The BARON/MATLAB interface was developed by J. Currie.  The interface
% is provided free of charge and with no warranties from The Optimization
% Firm, http://www.minlp.com.
% Interface version: v1.51 [17-Dec-2013]
% File generated: 06-Jun-2016 04:12:30
% Model: tranter2

% Objective
fun = @(x)(0.0211-x(1)-10*x(2)-100*x(3))^2+(0.0174-x(1)-10*x(2)-100*x(3))^2+(0.0329-x(1)-15*x(2)-225*x(3))^2+(0.0354-x(1)-20*x(2)-400*x(3))^2+(0.0462-x(1)-25*x(2)-625*x(3))^2+(0.0488-x(1)-30*x(2)-900*x(3))^2+(0.0788-x(1)-35*x(2)-1225*x(3))^2+(0.0675-x(1)-35*x(2)-1225*x(3))^2+(0.0818-x(1)-40*x(2)-1600*x(3))^2+(0.1054-x(1)-45*x(2)-2025*x(3))^2+(0.1251-x(1)-50*x(2)-2500*x(3))^2+(0.1588-x(1)-55*x(2)-3025*x(3))^2+(0.1829-x(1)-60*x(2)-3600*x(3))^2+(0.1782-x(1)-60*x(2)-3600*x(3))^2+(0.2101-x(1)-65*x(2)-4225*x(3))^2;

% Bounds
lb = [0,-1,0]';
ub = [0.1,0,0.1]';

% Constraints
nlcon = [];
cl = [];
cu = [];

% Variables (C = continuous, B = binary, I = integer)
xtype = 'CCC';

% Starting Guess
x0 = [NaN,NaN,NaN]';

% Options
opts = struct('probname','tranter2');
opts.sense = 'min';
