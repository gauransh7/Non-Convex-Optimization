% This BARON input file was generated by the MATLAB/BARON interface.
% The BARON/MATLAB interface was developed by J. Currie.  The interface
% is provided free of charge and with no warranties from The Optimization
% Firm, http://www.minlp.com.
% Interface version: v1.51 [17-Dec-2013]
% File generated: 06-Jun-2016 04:06:02
% Model: gridneta

% Objective
fun = @(x)0.01*(x(1)*x(1)+x(2)*x(2)+x(3)*x(3)+2*x(4)*x(4)+2*x(5)*x(5)+x(6)*x(6)+x(7)*x(7)+4*x(8)*x(8)+4*x(9)*x(9)+x(10)*x(10)+x(12)*x(12)+x(13)*x(13)+2*x(14)*x(14)+2*x(15)*x(15)+2*x(16)*x(16)+x(17)*x(17)+2*x(18)*x(18)+4*x(19)*x(19)+4*x(20)*x(20)+x(21)*x(21)+4*x(22)*x(22)+x(24)*x(24)+x(25)*x(25)+x(26)*x(26)+2*x(27)*x(27)+2*x(28)*x(28)+2*x(29)*x(29)+2*x(30)*x(30)+x(31)*x(31)+4*x(32)*x(32)+2*x(33)*x(33)+4*x(34)*x(34)+x(36)*x(36)+2*x(37)*x(37)+4*x(38)*x(38)+2*x(39)*x(39)+2*x(40)*x(40)+2*x(41)*x(41)+2*x(42)*x(42)+2*x(43)*x(43)+4*x(44)*x(44)+2*x(45)*x(45)+4*x(46)*x(46)+x(48)*x(48)+x(49)*x(49)+x(50)*x(50)+4*x(51)*x(51)+2*x(52)*x(52)+4*x(53)*x(53)+x(54)*x(54)+4*x(55)*x(55)+4*x(56)*x(56)+4*x(57)*x(57)+4*x(58)*x(58)+x(60)*x(60)+x(61)*x(61)+x(63)*x(63)+x(65)*x(65)+x(67)*x(67)+x(69)*x(69))+0.01*((1+x(1)^2+(x(1)-x(14))^2)^0.5+(1+x(2)^2+(x(2)-6*x(14))^2)^0.5+(1+x(3)^2+(x(3)-x(16))^2)^0.5+(1+x(4)^2+(x(4)-6*x(16))^2)^0.5+(1+x(5)^2+(x(5)-x(18))^2)^0.5+(1+x(6)^2+(x(6)-6*x(18))^2)^0.5+(1+x(7)^2+(x(7)-x(20))^2)^0.5+(1+x(8)^2+(x(8)-6*x(20))^2)^0.5+(1+x(9)^2+(x(9)-x(23))^2)^0.5+(1+x(10)^2+(x(10)-6*x(22))^2)^0.5+(1+x(12)^2+(x(12)-6*x(24))^2)^0.5+(1+x(13)^2+(x(13)-x(26))^2)^0.5+(1+x(14)^2+(x(14)-6*x(26))^2)^0.5+(1+x(15)^2+(x(15)-x(28))^2)^0.5+(1+x(16)^2+(x(16)-6*x(28))^2)^0.5+(1+x(17)^2+(x(17)-x(30))^2)^0.5+(1+x(18)^2+(x(18)-6*x(30))^2)^0.5+(1+x(19)^2+(x(19)-x(32))^2)^0.5+(1+x(20)^2+(x(20)-6*x(32))^2)^0.5+(1+x(21)^2+(x(21)-x(35))^2)^0.5+(1+x(22)^2+(x(22)-6*x(34))^2)^0.5+(1+x(24)^2+(x(24)-6*x(36))^2)^0.5+(1+x(25)^2+(x(25)-x(38))^2)^0.5+(1+x(26)^2+(x(26)-6*x(38))^2)^0.5+(1+x(27)^2+(x(27)-x(40))^2)^0.5+(1+x(28)^2+(x(28)-6*x(40))^2)^0.5+(1+x(29)^2+(x(29)-x(42))^2)^0.5+(1+x(30)^2+(x(30)-6*x(42))^2)^0.5+(1+x(31)^2+(x(31)-x(44))^2)^0.5+(1+x(32)^2+(x(32)-6*x(44))^2)^0.5+(1+x(33)^2+(x(33)-x(47))^2)^0.5+(1+x(34)^2+(x(34)-6*x(46))^2)^0.5+(1+x(36)^2+(x(36)-6*x(48))^2)^0.5+(1+x(37)^2+(x(37)-x(50))^2)^0.5+(1+x(38)^2+(x(38)-x(62))^2)^0.5+(1+x(39)^2+(x(39)-x(52))^2)^0.5+(1+x(40)^2+(x(40)-x(64))^2)^0.5+(1+x(41)^2+(x(41)-x(54))^2)^0.5+(1+x(42)^2+(x(42)-x(66))^2)^0.5+(1+x(43)^2+(x(43)-x(56))^2)^0.5+(1+x(44)^2+(x(44)-x(68))^2)^0.5+(1+x(45)^2+(x(45)-x(59))^2)^0.5+(1+x(46)^2+(x(46)-x(70))^2)^0.5+(1+x(48)^2+(x(48)-x(72))^2)^0.5+(1+x(49)^2+(x(49)-x(62))^2)^0.5+(1+x(50)^2+(x(50)-x(3))^2)^0.5+(1+x(51)^2+(x(51)-x(64))^2)^0.5+(1+x(52)^2+(x(52)-x(5))^2)^0.5+(1+x(53)^2+(x(53)-x(66))^2)^0.5+(1+x(54)^2+(x(54)-x(7))^2)^0.5+(1+x(55)^2+(x(55)-x(68))^2)^0.5+(1+x(56)^2+(x(56)-x(9))^2)^0.5+(1+x(57)^2+(x(57)-x(71))^2)^0.5+(1+x(58)^2+(x(58)-x(11))^2)^0.5+(1+x(60)^2+x(60)^2)^0.5+(1+x(61)^2+x(61)^2)^0.5+(1+x(63)^2+x(63)^2)^0.5+(1+x(65)^2+x(65)^2)^0.5+(1+x(67)^2+x(67)^2)^0.5+(1+x(69)^2+x(69)^2)^0.5)+0.000833333333333333*(10-x(1)+x(2)+x(3)-x(4)-x(5)+x(6)+x(7)-x(8)-x(9)+x(10)-x(13)+x(14)+x(15)-x(16)-x(17)+x(18)+x(19)-x(20)-x(21)+x(22)-x(25)+x(26)+x(27)-x(28)-x(29)+x(30)+x(31)-x(32)-x(33)+x(34)-x(37)+x(38)+x(39)-x(40)-x(41)+x(42)+x(43)-x(44)-x(45)+x(46)-x(49)+x(50)+x(51)-x(52)-x(53)+x(54)+x(55)-x(56)-x(57)+x(58))^4;

% Bounds
lb = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]';
ub = [Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf]';

% Constraints
nlcon = [];
cl = [];
cu = [];

% Variables (C = continuous, B = binary, I = integer)
xtype = 'CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC';

% Starting Guess
x0 = [NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN]';

% Options
opts = struct('probname','gridneta');
opts.sense = 'min';
