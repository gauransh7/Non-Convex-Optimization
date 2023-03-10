#include "ibex.h"

using namespace std;
using namespace ibex;

// lower bound on eigen value of Interval Matrix
double minEigenValueIntervalMatrix(IntervalMatrix &Im)
{
    int sz = Im.nb_rows();
    double min_eigen = POS_INFINITY;
    for(int i = 0 ; i < sz ; i++){
        int sum = 0;
        for(int j = 0 ; j < sz ; j++){
            if(j!=i){
                sum+=max(abs(Im[i][j].lb()),abs(Im[i][j].ub()));
            }
        }
        min_eigen = min(min_eigen, Im[i][i].lb()-sum);
    }
    return min_eigen;
}

int main(int argc, char **argv)
{
    time_t start, end;
  
    /* You can call it like this : start = time(NULL);
     in both the way start contain total time in seconds 
     since the Epoch. */
    time(&start);
  
    // unsync the I/O of C and C++.
    ios_base::sync_with_stdio(false);
    string func = "exp(0.1*x(1)*x(2))+exp(0.1*x(2)*x(3))+exp(0.1*x(3)*x(4))+exp(0.1*x(4)*x(5))+exp(0.1*x(5)*x(6))+exp(0.1*x(6)*x(7))+exp(0.1*x(7)*x(8))+exp(0.1*x(8)*x(9))+exp(0.1*x(9)*x(10))+exp(0.1*x(10)*x(11))-10*x(1)-20*x(2)-30*x(3)-40*x(4)-50*x(5)-60*x(6)-70*x(7)-80*x(8)-90*x(9)-100*x(10)-110*x(11)-120*x(12)-130*x(13)-140*x(14)-150*x(15)-160*x(16)-170*x(17)-180*x(18)-190*x(19)-200*x(20)-210*x(21)-220*x(22)-230*x(23)-240*x(24)-250*x(25)-260*x(26)-270*x(27)-280*x(28)-290*x(29)-300*x(30)-310*x(31)-320*x(32)-330*x(33)-340*x(34)-350*x(35)-360*x(36)-370*x(37)-380*x(38)-390*x(39)-400*x(40)-410*x(41)-420*x(42)-430*x(43)-440*x(44)-450*x(45)-460*x(46)-470*x(47)-480*x(48)-490*x(49)-500*x(50)-510*x(51)-520*x(52)-530*x(53)-540*x(54)-550*x(55)-560*x(56)-570*x(57)-580*x(58)-590*x(59)-600*x(60)-610*x(61)-620*x(62)-630*x(63)-640*x(64)-650*x(65)-660*x(66)-670*x(67)-680*x(68)-690*x(69)-700*x(70)-710*x(71)-720*x(72)-730*x(73)-740*x(74)-750*x(75)-760*x(76)-770*x(77)-780*x(78)-790*x(79)-800*x(80)-810*x(81)-820*x(82)-830*x(83)-840*x(84)-850*x(85)-860*x(86)-870*x(87)-880*x(88)-890*x(89)-900*x(90)-910*x(91)-920*x(92)-930*x(93)-940*x(94)-950*x(95)-960*x(96)-970*x(97)-980*x(98)-990*x(99)-1000*x(100)-1010*x(101)-1020*x(102)-1030*x(103)-1040*x(104)-1050*x(105)-1060*x(106)-1070*x(107)-1080*x(108)-1090*x(109)-1100*x(110)-1110*x(111)-1120*x(112)-1130*x(113)-1140*x(114)-1150*x(115)-1160*x(116)-1170*x(117)-1180*x(118)-1190*x(119)-1200*x(120)";
    // string func = "x(1)^2 + x(2)^2";
    const char* func_string = func.c_str();
    cout << "Function: \n" << func << endl;
    Function f("x[120]",func_string);
    Function df(f,Function::DIFF);

    double _x[120][2] = {{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7},{4,7}};

    IntervalVector xy(120,_x); // build xy=([1,2],[3,4])
    // xy[0] = xy[1] = xy[2] = xy[3] = Interval(1,10);
    Interval z = f.eval(xy);  // z=f(xy)=sin([4,6])=[-1, -0.27941]

    cout << "Input interval: \n" << xy << "\nOutput Interval : " << z << endl;
    // Recording end time.
    time(&end);
  
    // Calculating total time taken by the program.
    double time_taken = double(end - start);
    cout << "Time taken by program is : " << fixed
         << time_taken << setprecision(5);
    cout << "sec " << endl;

}
