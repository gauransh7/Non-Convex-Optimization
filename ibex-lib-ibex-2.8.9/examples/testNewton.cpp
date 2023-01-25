// import libraries
#include "ibex.h"
#include "eigen/Eigen/Dense"
#include "eigen/Eigen/QR"

using namespace std;
using namespace ibex;

// this function convert std c++ vector to Eigen vector
Eigen::VectorXd ConvertToEigenVector(vector<double> v)
{
    double *ptr_data = &v[0];
    Eigen::VectorXd v2 = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(v.data(), v.size());
    return v2;
}
// this function convert Eigen vector to std c++ vector
vector<double> ConvertTo1dVector(Eigen::VectorXd mat)
{
    vector<double> vec(mat.data(), mat.data() + mat.rows() * mat.cols());
    return vec;
}

// this function convert Eigen Matrix to std c++ 2d vector
vector<vector<double>> ConvertTo2dVector(Eigen::MatrixXd m)
{

    std::vector<std::vector<double>> v;

    for (int i = 0; i < m.rows(); ++i)
    {
        const double *begin = &m.row(i).data()[0];
        v.push_back(std::vector<double>(begin, begin + m.cols()));
    }

    return v;
}

vector<vector<double>> ConvertIbexMatrixTo2DVector(ibex::Matrix m){
    int n = m.nb_rows();
    vector<vector<double>>dummyHessian(n, vector<double>(n));
    for(int i =0 ; i < n ; i++){
        for(int j = 0 ; j < n ; j++){
            dummyHessian[i][j] = m[i][j];
        }
    }
    return dummyHessian;
}

// this function convert std c++ 2d vector to Eigen Matrix
Eigen::MatrixXd ConvertToEigenMatrix(std::vector<std::vector<double>> data)
{
    Eigen::MatrixXd eMatrix(data.size(), data[0].size());
    for (int i = 0; i < data.size(); ++i)
        eMatrix.row(i) = Eigen::VectorXd::Map(&data[i][0], data[0].size());
    return eMatrix;
}

// this function return norm of vector (L2)
double normOfVector(Vector old_vec, Vector new_vec, int n, int iter)
{
    if (iter == 0)
        return 1;
    double gr = 0;
    for (int i = 0; i < n; i++)
    {
        gr += ((new_vec[i] - old_vec[i]) * (new_vec[i] - old_vec[i]));
    }
    cout << "Norm : " << sqrt(gr) << endl;
    return sqrt(gr);
}

// Returs Direction Vector from hessian and gradient
Vector DirectionVector(Matrix hessian, Vector grad)
{
    int n = grad.size();

    Vector direcV(Vector ::zeros(n));
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            direcV[i] = hessian[i][j] * grad[j];
        }
    }
    // double nrm = norm(direcV);
    // for(int i = 0 ; i < n ; i++){
    //     direcV[i] = direcV[i]/nrm;
    // }
    return direcV;
}

// Returns Gradient(double) from gradient func and point
Vector gradVector(Function gradient, Vector xk)
{
    int n = xk.size();

    Vector gradV(n);
    for (int i = 0; i < n; i++)
    {
        IntervalVector result = gradient[i].eval(xk);
        gradV[i] = result.lb()[0];
    }
    // double nrm = norm(gradV);
    // for(int i = 0 ; i < n ; i++){
    //     gradV[i] = gradV[i]/nrm;
    // }
    return gradV;
}
// Returns Hessian(double) from gradient func and point
Matrix HessianMatrix(Function gradient, Vector xk)
{
    int n = xk.size();
    Matrix hessian(n, n);
    for (int i = 0; i < n; i++)
    {
        Function new_f(gradient[i], Function::DIFF);
        for (int j = i; j < n; j++)
        {
            IntervalVector result = new_f[i].eval(xk);
            hessian[i][j] = hessian[j][i] = result.lb()[0];
        }
    }
    return hessian;
}

// lower bound on eigen value of Interval Matrix
double minEigenValueIntervalMatrix(IntervalMatrix &Im)
{
    int sz = Im.nb_rows();
    double min_eigen = POS_INFINITY;
    for (int i = 0; i < sz; i++)
    {
        int sum = 0;
        for (int j = 0; j < sz; j++)
        {
            if (j != i)
            {
                sum += max(abs(Im[i][j].lb()), abs(Im[i][j].ub()));
            }
        }
        min_eigen = min(min_eigen, Im[i][i].lb() - sum);
    }
    return min_eigen;
}

bool checkCholesky(Matrix hessian){
    vector<vector<double>> dummyHessian = ConvertIbexMatrixTo2DVector(hessian);
    Eigen::LLT<Eigen::MatrixXd> lltOfA(ConvertToEigenMatrix(dummyHessian)); // compute the Cholesky decomposition of Hessian
    Eigen::ComputationInfo m_info = lltOfA.info();
    cout << "Cholesky Results : " << m_info << endl;
    if (m_info == 1)
    {
        cout << "---Encountered a Non-Postive Definite Matrix.---" << endl;
        return true;
        // return 0;
    }
    else{
        return false;
    }
}

Matrix ModifyHessian(string func, string fnc_param, Vector xk, double alp, Matrix hessian)
{
    int m = xk.size();
    // Matrix hessian(m, m);
    const char *func_string = func.c_str();
    cout << "Non Convex Function: \n"
         << func << endl;
    Function f(fnc_param.c_str(), func_string);
    Function df(f, Function::DIFF);

    // Interval Calculation [xk-alp,xk+alp]
    double _x[m][2];
    for (int i = 0; i < m; i++)
    {
        _x[i][0] = xk[i] - alp;
        _x[i][1] = xk[i] + alp;
    }

    IntervalVector xy(m, _x); // build xy=([1,2],[3,4])
    IntervalMatrix im(m, m);
    for (int i = 0; i < m; i++)
    {
        cout << "Differentiating df(" << i + 1 << ")\n";
        Function new_f(df[i], Function::DIFF);
        for (int j = i; j < m; j++)
        {
            im[i][j] = im[j][i] = new_f[j].eval(xy);
        }
    }
    cout << "Lower bound on Minimum Eigen Value:\n";
    double lowerBoundEigenValue = minEigenValueIntervalMatrix(im);
    if(lowerBoundEigenValue<-100000){
        lowerBoundEigenValue = -5000;
    }
    cout << lowerBoundEigenValue << endl;
    double alpha = max(double(0), -0.5 * lowerBoundEigenValue);

    hessian = hessian + 2*alpha*(Matrix::eye(xk.size()));
    return hessian;
}

int main(int argc, char **argv)
{
    int n = 1;
    string fnc_param = "x[" + to_string(n) + "]";
    // string func = "sin(x(1))+(2.71^sin(x(2)))*x(2)";
    // string func = "sin(x(1))";
    // string func = "x(1)*x(1)*x(1)";
    
    // visualization
    string func = "x(1)*(2.71^sin(x(1)))";

    // alpine1
    // string func = "abs(x(1) *sin(x(1))+ 0.1 * x(1))";
    
    // amgm2
    // string func = "1/2 * (x(1)+x(2))- (x(1)*x(2))^0.5";

    // 5 variable basic
    // string func = "x(1)*x(1)*x(1)*x(1)*x(1)+x(2)*x(2)*x(2)*x(2)+x(3)*x(3)*x(3)+x(4)*x(4)+x(5)";
    

    // alpine5
    // string func = "abs(x(1) *sin(x(1))+ 0.1 * x(1)) + abs(x(2) *sin(x(2))+ 0.1 * x(2)) + abs(x(3) *sin(x(3))+ 0.1 * x(3)) + abs(x(4) *sin(x(4))+ 0.1 * x(4)) +  abs(x(5) *sin(x(5))+ 0.1 * x(5))";
    
    // ackley 5
    // string func = "-20*exp(-0.2*sqrt(0.2*(x(1)^2+x(2)^2+x(3)^2+x(4)^2+x(5)^2)))-exp(0.2*(cos(2*pi*x(1))+cos(2*pi*x(2))+cos(2*pi*x(3))+cos(2*pi*x(4))+cos(2*pi*x(5))))+20+2.718281";
    const char *func_string = func.c_str();
    cout << "Non Convex Function: \n"
         << func << endl;
    Function f(fnc_param.c_str(), func_string);
    Function df(f, Function::DIFF);

    Matrix hessian(n, n);
    Vector xk(n), xkn(n);
    cout << "Initial guess: ";
    for (int i = 0; i < n; i++)
    {
        cin >> xkn[i];
    }
    Vector grad(n), direcV(n), grad_new(n);
    double a_init = 1;
    double a = a_init;

    // c constant used for check strong wolf condition
    double c = 0.8;

    double a_min = pow(10, -6);

    // rho value to update alpha value
    double rh = 0.1;
    int iter = 1;
    IntervalVector result, prevResult;
    while (true)
    {
        double norm_vec = normOfVector(xk, xkn, n, iter);
        cout << "norm : " << norm_vec << endl;
        if(iter>1) prevResult = result;
        result = f.eval(xkn);
        cout << "Function Value : " << result.lb()[0] << endl;
        if (norm_vec < 1e-6)
        {
            cout << "Result: Success" << endl;
            break;
        }
        cout << "starting iter: " << iter << endl;
        iter++;
        cout << "New xk : " << endl;
        for (int i = 0; i < n; i++)
        {
            xk[i] = xkn[i];
            cout << xk[i] << " ";
        }
        cout << "\n";
        hessian = HessianMatrix(df, xk);
        cout << "---Computed Hessian---" << endl;
        bool cholresult = checkCholesky(hessian);
        grad = gradVector(df, xk);
        cout << "---Computed Gradient---" << endl;
        for (int i = 0; i < n; i++)
        {
            grad[i] = -grad[i];
        }
        direcV = DirectionVector(hessian, grad);
        cout << "---Computed Direction Vector---" << endl;
        for (int i = 0; i < n; i++)
        {
            grad[i] = -grad[i];
        }
        double gfpk = 0.0000;
        for (int i = 0; i < n; i++)
        {
            gfpk += direcV[i] * grad[i];
        }
        if(gfpk>0){
            a = 1;
            cout << "Modifieng Hessian" << endl;
            hessian = ModifyHessian(func, fnc_param, xk, 1.0, hessian);
            cholresult = checkCholesky(hessian);
            for (int i = 0; i < n; i++)
            {
                grad[i] = -grad[i];
            }
            direcV = DirectionVector(hessian, grad);
            cout << "---Computed Direction Vector---" << endl;
            for (int i = 0; i < n; i++)
            {
                grad[i] = -grad[i];
            }
            gfpk = 0.0000;
            for (int i = 0; i < n; i++)
            {
                gfpk += direcV[i] * grad[i];
            }
        }

        for (int i = 0; i < n; i++)
        {
            xkn[i] = xk[i] + a * direcV[i];
        }
        IntervalVector func_value_xkn = f.eval(xkn);
        while(func_value_xkn.lb()[0] > result.lb()[0] + c*a*gfpk){
            a = a*rh;
            if(gfpk>0 || a<a_min){
                a = 1;
                cout << "Modifieng Hessian" << endl;
                hessian = ModifyHessian(func, fnc_param, xk, 1.0, hessian);
                cholresult = checkCholesky(hessian);
                for (int i = 0; i < n; i++)
                {
                    grad[i] = -grad[i];
                }
                direcV = DirectionVector(hessian, grad);
                cout << "---Computed Direction Vector---" << endl;
                for (int i = 0; i < n; i++)
                {
                    grad[i] = -grad[i];
                }
                gfpk = 0.0000;
                for (int i = 0; i < n; i++)
                {
                    gfpk += direcV[i] * grad[i];
                }
                for (int i = 0; i < n; i++)
                {
                    xkn[i] = xk[i] + a * direcV[i];
                }
                func_value_xkn = f.eval(xkn);
            }
            cout << "a : " << a << endl; 
            for (int i = 0; i < n; i++)
            {
                xkn[i] = xk[i] + a * direcV[i];
            }
            func_value_xkn = f.eval(xkn);
            cout << "loop results:" << func_value_xkn.lb()[0] << " " << result.lb()[0]  << " " << c*a*gfpk << endl;
        }
        a = a_init;
    }
    cout << "Iterations: " << iter << endl;
    // this print our final x*(point of minima)
    for (int i = 0; i < n; i++)
    {
        cout << fixed << setprecision(7) << xkn[i] << " ";
    }
    cout << endl;
    return 0;
}
