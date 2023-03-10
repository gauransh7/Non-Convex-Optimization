% This BARON input file was generated by the MATLAB/BARON interface.
% The BARON/MATLAB interface was developed by J. Currie.  The interface
% is provided free of charge and with no warranties from The Optimization
% Firm, http://www.minlp.com.
% Interface version: v1.51 [17-Dec-2013]
% File generated: 06-Jun-2016 04:12:10
% Model: s292

% Objective
fun = @(x)x(1)*x(1)+2*x(2)*x(2)+3*x(3)*x(3)+4*x(4)*x(4)+5*x(5)*x(5)+6*x(6)*x(6)+7*x(7)*x(7)+8*x(8)*x(8)+9*x(9)*x(9)+10*x(10)*x(10)+11*x(11)*x(11)+12*x(12)*x(12)+13*x(13)*x(13)+14*x(14)*x(14)+15*x(15)*x(15)+16*x(16)*x(16)+17*x(17)*x(17)+18*x(18)*x(18)+19*x(19)*x(19)+20*x(20)*x(20)+21*x(21)*x(21)+22*x(22)*x(22)+23*x(23)*x(23)+24*x(24)*x(24)+25*x(25)*x(25)+26*x(26)*x(26)+27*x(27)*x(27)+28*x(28)*x(28)+29*x(29)*x(29)+30*x(30)*x(30);

% Bounds
lb = [-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf]';
ub = [Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf]';

% Constraints
nlcon = [];
cl = [];
cu = [];

% Variables (C = continuous, B = binary, I = integer)
xtype = 'CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC';

% Starting Guess
x0 = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]';

% Options
opts = struct('probname','s292');
opts.sense = 'min';
