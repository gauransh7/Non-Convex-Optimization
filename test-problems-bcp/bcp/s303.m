% This BARON input file was generated by the MATLAB/BARON interface.
% The BARON/MATLAB interface was developed by J. Currie.  The interface
% is provided free of charge and with no warranties from The Optimization
% Firm, http://www.minlp.com.
% Interface version: v1.51 [17-Dec-2013]
% File generated: 06-Jun-2016 04:12:10
% Model: s303

% Objective
fun = @(x)x(1)^2+x(2)^2+x(3)^2+x(4)^2+x(5)^2+x(6)^2+x(7)^2+x(8)^2+x(9)^2+x(10)^2+x(11)^2+x(12)^2+x(13)^2+x(14)^2+x(15)^2+x(16)^2+x(17)^2+x(18)^2+x(19)^2+x(20)^2+(0.5*x(1)+x(2)+1.5*x(3)+2*x(4)+2.5*x(5)+3*x(6)+3.5*x(7)+4*x(8)+4.5*x(9)+5*x(10)+5.5*x(11)+6*x(12)+6.5*x(13)+7*x(14)+7.5*x(15)+8*x(16)+8.5*x(17)+9*x(18)+9.5*x(19)+10*x(20))^2+(0.5*x(1)+x(2)+1.5*x(3)+2*x(4)+2.5*x(5)+3*x(6)+3.5*x(7)+4*x(8)+4.5*x(9)+5*x(10)+5.5*x(11)+6*x(12)+6.5*x(13)+7*x(14)+7.5*x(15)+8*x(16)+8.5*x(17)+9*x(18)+9.5*x(19)+10*x(20))^4;

% Bounds
lb = [-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf]';
ub = [Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf]';

% Constraints
nlcon = [];
cl = [];
cu = [];

% Variables (C = continuous, B = binary, I = integer)
xtype = 'CCCCCCCCCCCCCCCCCCCC';

% Starting Guess
x0 = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]';

% Options
opts = struct('probname','s303');
opts.sense = 'min';
