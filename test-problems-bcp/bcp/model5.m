% This BARON input file was generated by the MATLAB/BARON interface.
% The BARON/MATLAB interface was developed by J. Currie.  The interface
% is provided free of charge and with no warranties from The Optimization
% Firm, http://www.minlp.com.
% Interface version: v1.51 [17-Dec-2013]
% File generated: 06-Jun-2016 04:10:05
% Model: model5

% Objective
fun = @(x)(34780-exp(x(2)/(50+x(3)))-x(1))^2+(28610-exp(x(2)/(55+x(3)))-x(1))^2+(23650-exp(x(2)/(60+x(3)))-x(1))^2+(19630-exp(x(2)/(65+x(3)))-x(1))^2+(16370-exp(x(2)/(70+x(3)))-x(1))^2+(13720-exp(x(2)/(75+x(3)))-x(1))^2+(11540-exp(x(2)/(80+x(3)))-x(1))^2+(9744-exp(x(2)/(85+x(3)))-x(1))^2+(8261-exp(x(2)/(90+x(3)))-x(1))^2+(7030-exp(x(2)/(95+x(3)))-x(1))^2+(6005-exp(x(2)/(100+x(3)))-x(1))^2+(5147-exp(x(2)/(105+x(3)))-x(1))^2+(4427-exp(x(2)/(110+x(3)))-x(1))^2+(3820-exp(x(2)/(115+x(3)))-x(1))^2+(3307-exp(x(2)/(120+x(3)))-x(1))^2+(2872-exp(x(2)/(125+x(3)))-x(1))^2;

% Bounds
lb = [0,0,0]';
ub = [10,1100,500]';

% Constraints
nlcon = [];
cl = [];
cu = [];

% Variables (C = continuous, B = binary, I = integer)
xtype = 'CCC';

% Starting Guess
x0 = [NaN,NaN,NaN]';

% Options
opts = struct('probname','model5');
opts.sense = 'min';
