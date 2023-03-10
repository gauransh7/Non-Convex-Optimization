// import libraries
#include <iostream>
#include <fstream>
#include <string>
#include "ibex.h"
// changed g++ flag to -std=c++17 to support eigen
#include "eigen/Eigen/Dense"
#include "eigen/Eigen/QR"

using namespace std;
using namespace ibex;

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
            direcV[i] += hessian[i][j] * grad[j];
        }
    }
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
    return gradV;
}
// Returns Hessian(double) from gradient func and point
Matrix HessianMatrix(Function dff, Vector xk)
{
    int n = xk.size();
    Matrix hessian(n, n);
    for (int i = 0; i < n; i++)
    {
        // Function new_f(gradient[i], Function::DIFF);
        for (int j = i; j < n; j++)
        {
            IntervalVector result = dff[i][j].eval(xk);
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

Matrix CalculateModifiedMidPointMatrix(IntervalMatrix &Im)
{
    int sz = Im.nb_rows();
    Matrix ModifiedMatrix(sz,sz);
    for(int i = 0 ; i < sz ; i++){
        for(int j = 0 ; j < sz ; j++){
            if(i==j){
                ModifiedMatrix[i][j] = Im[i][j].lb();
            }
            else{
                ModifiedMatrix[i][j] = (Im[i][j].lb() + Im[i][j].ub())/2;
            }
        }
    }
    return ModifiedMatrix;
}

Matrix CalculateEMatrix(IntervalMatrix &Im)
{
    int sz = Im.nb_rows();
    Matrix ModifiedMatrix(sz,sz);
    for(int i = 0 ; i < sz ; i++){
        for(int j = 0 ; j < sz ; j++){
            if(i==j){
                ModifiedMatrix[i][j] = (Im[i][j].ub() - Im[i][j].lb())/2;
            }
            else{
                ModifiedMatrix[i][j] = 0;
            }
        }
    }
    return ModifiedMatrix;
}

Matrix CalculateModifiedRadiusMatrix(IntervalMatrix &Im)
{
    int sz = Im.nb_rows();
    Matrix ModifiedMatrix(sz,sz);
    for(int i = 0 ; i < sz ; i++){
        for(int j = 0 ; j < sz ; j++){
            if(i==j){
                ModifiedMatrix[i][j] = 0;
            }
            else{
                ModifiedMatrix[i][j] = (Im[i][j].ub() - Im[i][j].lb())/2;
            }
        }
    }
    return ModifiedMatrix;
}

double calcLowerBoundEigenValue(Matrix mt)
{
    int sz = mt.nb_rows();
    double min_eigen = POS_INFINITY;
    for (int i = 0; i < sz; i++)
    {
        int sum = 0;
        for (int j = 0; j < sz; j++)
        {
            if (j != i)
            {
                sum += abs(mt[i][j]);
            }
        }
        min_eigen = min(min_eigen, mt[i][i] - sum);
    }
    return min_eigen;
}

double calcUpperBoundEigenValue(Matrix mt)
{
    int sz = mt.nb_rows();
    double max_eigen = POS_INFINITY;
    for (int i = 0; i < sz; i++)
    {
        int sum = 0;
        for (int j = 0; j < sz; j++)
        {
            if (j != i)
            {
                sum += abs(mt[i][j]);
            }
        }
        max_eigen = max(max_eigen, mt[i][i] + sum);
    }
    return max_eigen;
}

double spectrum(Matrix mt)
{
    double lower_bound = calcLowerBoundEigenValue(mt);
    double upper_bound = calcUpperBoundEigenValue(mt);

    return max(abs(lower_bound),abs(upper_bound));
}

double On3MinEigenValueIntervalMatrix(IntervalMatrix &Im)
{
    Matrix ModifiedMPMatrix = CalculateModifiedMidPointMatrix(Im);
    Matrix EMatrix = CalculateEMatrix(Im);
    Matrix ModifiedRadiusMatrix = CalculateModifiedRadiusMatrix(Im);
    double sp = spectrum(ModifiedRadiusMatrix+EMatrix);
    double lb = calcLowerBoundEigenValue(ModifiedMPMatrix+EMatrix);

    return lb-sp;
}

bool checkCholesky(Matrix hessian){
    vector<vector<double>> dummyHessian = ConvertIbexMatrixTo2DVector(hessian);
    Eigen::LLT<Eigen::MatrixXd> lltOfA(ConvertToEigenMatrix(dummyHessian)); // compute the Cholesky decomposition of Hessian
    Eigen::ComputationInfo m_info = lltOfA.info();
    if (m_info == 1)
    {
        return true;
        // return 0;
    }
    else{
        return false;
    }
}

Matrix ModifyHessianDS(Matrix hessian)
{
    int n = hessian.nb_rows();
    Matrix IDMatrix = Matrix::eye(n);
    int cnt = 0;
    while(checkCholesky(hessian) && cnt<1000){
        cnt++;
        hessian = hessian+IDMatrix;
    }
    return hessian;
}

Matrix ModifyHessianGerschgorin(Matrix hessian){
    double min_eigen = POS_INFINITY;
    int n = hessian.nb_rows();
    for(int i = 0 ; i < n ; i++){
        double sum = 0;
        for(int j = 0 ; j < n ; j++){
            if(j!=i || j==0){
                sum += abs(hessian[i][j]);
            }
        }
        min_eigen = min(min_eigen, hessian[i][i]-sum);
    }

    if(min_eigen<0){
        hessian = hessian + (-1*min_eigen)*(Matrix::eye(n));
    }

    return hessian;

}

Matrix FindTestVectors(int p, int n){
    return ibex::Matrix::rand(p,n);
}

Matrix ModifyHessianOn3(string func, string fnc_param, Vector xk, double alp, Matrix hessian, Function dff)
{
    int m = xk.size();
    // Matrix hessian(m, m);
    // const char *func_string = func.c_str();
    // Function f(fnc_param.c_str(), func_string);
    // Function df(f, Function::DIFF);

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
        // Function new_f(df[i], Function::DIFF);
        for (int j = i; j < m; j++)
        {
            im[i][j] = im[j][i] = dff[i][j].eval(xy);
        }
    }
    double lowerBoundEigenValue = On3MinEigenValueIntervalMatrix(im);
    if(lowerBoundEigenValue<-100000){
        lowerBoundEigenValue = -5000;
    }
    double alpha = max(double(0), -0.5 * lowerBoundEigenValue);

    hessian = hessian + 2*alpha*(Matrix::eye(xk.size()));
    return hessian;
}

Matrix ModifyHessianOn2(string func, string fnc_param, Vector xk, double alp, Matrix hessian, Function dff)
{
    int m = xk.size();
    // Matrix hessian(m, m);
    // const char *func_string = func.c_str();
    // Function f(fnc_param.c_str(), func_string);
    // Function df(f, Function::DIFF);

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
        // Function new_f(df[i], Function::DIFF);
        for (int j = i; j < m; j++)
        {
            im[i][j] = im[j][i] = dff[i][j].eval(xy);
        }
    }
    double lowerBoundEigenValue = minEigenValueIntervalMatrix(im);
    if(lowerBoundEigenValue<-100000){
        lowerBoundEigenValue = -5000;
    }
    double alpha = max(double(0), -0.5 * lowerBoundEigenValue);

    hessian = hessian + 2*alpha*(Matrix::eye(xk.size()));
    return hessian;
}

double random(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

int main(int argc, char **argv)
{
    fstream test_problems;

    // open the dummyProblems file to perform read operation using file object.
    test_problems.open("on3test.txt", ios::in); 

    if (!test_problems.is_open()){
        cout << "Unable to open dummyProblems file." << endl;
        return 0;
    }

    double alp_options[5] = {0.01,0.1,0.5,1};

    string problem_name;
    while(getline(test_problems, problem_name)){
        fstream fout;

        problem_name = problem_name.substr(0,problem_name.length()-1);
        string filename = problem_name+".csv";
        fout.open(filename, std::ofstream::out | std::ofstream::trunc);
        
        fout << "IterationGG" << "," << "IterationON2" << "," << "IterationDS" << "," << "IterationON3" << "," << "FuncValueGG" << "," << "FuncValueON2" << "," << "FuncValueDS" << "," << "FuncValueON3" << "," << "Alp" << "\n";
        string n_str, range_str;
        int n;
        double range_ll, range_ul;
        string func;
        cout << "Input n: ";
        // cin >> n;
        getline(test_problems, n_str);
        n = stoi(n_str);
        cout << "Input Function (String): ";
        getline(test_problems, func);
        // cin >> func;
        string fnc_param = "x[" + to_string(n) + "]";
        cout << "Initial guess range: ";
        getline(test_problems, range_str);
        range_ll = stod(range_str);
        getline(test_problems, range_str);
        range_ul = stod(range_str);
        int p_max = 300;
        // cout << "No. of guess : ";
        // cin >> p_max;
        // cin >> range_ll >> range_ul;

        const char *func_string = func.c_str();
        cout << "Non Convex Function: \n"
            << func << endl;
        Function f(fnc_param.c_str(), func_string);
        Function df(f, Function::DIFF);
        Function dff(df, Function::DIFF);

        Matrix hessian(n, n);
        Vector xk(n), xkn(n), inixk(n);
        Vector fs_our(n), fs_ger(n);
        Matrix test_vectors = FindTestVectors(500,n);
        double func_value_ger, func_value_on2, func_value_ds, func_value_on3;
        int our_better = 0, total_test = 0, func_value_better = 0;
        for(int p = 0 ; p < p_max ; p++){
            for(int j = 0 ; j < n ; j++){
                inixk[j] = random(range_ll, range_ul);
            }
            xkn = inixk;
            // for (int i = 0; i < n; i++)
            // {
            //     cin >> xkn[i];
            //     inixk[i] = xkn[i];
            // }
            cout << "Executing " << p << " initial guess.\n";
            for (int i = 0; i < n; i++)
            {
                cout << inixk[i] << " ";
            }
            cout << endl;
            Vector grad(n), direcV(n), grad_new(n);
            double a_init = 1;
            double a = a_init;

            // c constant used for check strong wolf condition
            double c = 0.8;

            double a_min = pow(10, -20);

            // rho value to update alpha value
            double rh = 0.01;
            int iter = 1;
            int iter_max = 20000;
            int hessian_modify_count = 0;
            IntervalVector result, prevResult;
            while (true)
            {
                if(hessian_modify_count==0 && iter>100){
                    break;
                }
                if(iter>iter_max){
                    break;
                }
                double norm_vec = normOfVector(xk, xkn, n, iter);
                if(iter>1) prevResult = result;
                result = f.eval(xkn);
                if (norm_vec < 1e-6)
                {
                    break;
                }
                iter++;
                for (int i = 0; i < n; i++)
                {
                    xk[i] = xkn[i];
                }
                hessian = HessianMatrix(dff, xk);
                // bool cholresult = checkCholesky(hessian);
                grad = gradVector(df, xk);
                for (int i = 0; i < n; i++)
                {
                    grad[i] = -grad[i];
                }
                direcV = DirectionVector(hessian, grad);
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
                    hessian_modify_count+=1;
                    hessian = ModifyHessianGerschgorin(hessian);
                    // cholresult = checkCholesky(hessian);
                    for (int i = 0; i < n; i++)
                    {
                        grad[i] = -grad[i];
                    }
                    direcV = DirectionVector(hessian, grad);
                    for (int i = 0; i < n; i++)
                    {
                        grad[i] = -grad[i];
                    }
                    gfpk = 0.0000;
                    for (int i = 0; i < n; i++)
                    {
                        gfpk += direcV[i] * grad[i];
                    }
                    if(gfpk>0){
                        break;
                    }
                }

                for (int i = 0; i < n; i++)
                {
                    xkn[i] = xk[i] + a * direcV[i];
                }
                IntervalVector func_value_xkn = f.eval(xkn);
                while(func_value_xkn.lb()[0] > result.lb()[0] + c*a*gfpk){
                    a = a*rh;
                    if(a<a_min){
                        break;
                    }

                    for (int i = 0; i < n; i++)
                    {
                        xkn[i] = xk[i] + a * direcV[i];
                    }
                    func_value_xkn = f.eval(xkn);
                }
                if(a<a_min){
                    break;
                }
                a = a_init;
                // fout << iter << "," << func_value_xkn.lb()[0] << "\n";
                func_value_ger = func_value_xkn.lb()[0];
            }
            fs_our = xkn;
            if(hessian_modify_count==0){
                cout << "Hessian was not modified\n";
                continue;
            }
            cout << "GerschGorin Approach\n";
            std :: cout << "Iterations: " << iter << " & Hessian Modified " << hessian_modify_count << " times." << endl;
            
            // this print our final x*(point of minima)
            int ger_iter = iter;
            if(ger_iter>=iter_max){
                continue;
            }
            total_test++;
            for (int i = 0; i < n; i++)
            {
                std :: cout << fixed << setprecision(7) << xkn[i] << " ";
            }
            std :: cout << endl;


            a = a_init;
            iter = 1;
            xkn = inixk;
            hessian_modify_count = 0;
            cout << "DS Approach\n";
            bool unable_DS = false;
            while (true)
            {
                if(hessian_modify_count==0 && iter>100){
                    break;
                }
                if(iter>iter_max){
                    break;
                }
                double norm_vec = normOfVector(xk, xkn, n, iter);
                if(iter>1) prevResult = result;
                result = f.eval(xkn);
                if (norm_vec < 1e-6)
                {
                    break;
                }
                iter++;
                for (int i = 0; i < n; i++)
                {
                    xk[i] = xkn[i];
                }
                hessian = HessianMatrix(dff, xk);
                // bool cholresult = checkCholesky(hessian);
                grad = gradVector(df, xk);
                for (int i = 0; i < n; i++)
                {
                    grad[i] = -grad[i];
                }
                direcV = DirectionVector(hessian, grad);
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
                    hessian_modify_count+=1;
                    hessian = ModifyHessianDS(hessian);
                    if(checkCholesky(hessian)){
                        unable_DS=true;
                        break;
                    }
                    for (int i = 0; i < n; i++)
                    {
                        grad[i] = -grad[i];
                    }
                    direcV = DirectionVector(hessian, grad);
                    for (int i = 0; i < n; i++)
                    {
                        grad[i] = -grad[i];
                    }
                    gfpk = 0.0000;
                    for (int i = 0; i < n; i++)
                    {
                        gfpk += direcV[i] * grad[i];
                    }
                    if(gfpk>0){
                        break;
                    }
                }

                for (int i = 0; i < n; i++)
                {
                    xkn[i] = xk[i] + a * direcV[i];
                }
                IntervalVector func_value_xkn = f.eval(xkn);
                while(func_value_xkn.lb()[0] > result.lb()[0] + c*a*gfpk){
                    a = a*rh;
                    if(a<a_min){
                        break;
                    }

                    for (int i = 0; i < n; i++)
                    {
                        xkn[i] = xk[i] + a * direcV[i];
                    }
                    func_value_xkn = f.eval(xkn);
                }
                if(a<a_min){
                    break;
                }
                a = a_init;
                // fout << iter << "," << func_value_xkn.lb()[0] << "\n";
                func_value_ds = func_value_xkn.lb()[0];
            }
            fs_our = xkn;
            if(hessian_modify_count==0){
                cout << "Hessian was not modified\n";
                continue;
            }
            cout << "DS Approach\n";
            std :: cout << "Iterations: " << iter << " & Hessian Modified " << hessian_modify_count << " times." << endl;
            if(unable_DS){
                cout << "Hessian took more than 1000 iterations to modify" << endl;
                continue;
            }
            // this print our final x*(point of minima)
            int ds_iter = iter;
            for (int i = 0; i < n; i++)
            {
                std :: cout << fixed << setprecision(7) << xkn[i] << " ";
            }
            std :: cout << endl;

            for(int i = 0 ; i < sizeof(alp_options)/sizeof(alp_options[0]) ; i++)
            {
                a = a_init;
                iter = 1;
                xkn = inixk;
                hessian_modify_count = 0;
                cout << "ON2 Approach\n";
                while (true)
                {
                    if(iter>iter_max){
                        break;
                    }
                    double norm_vec = normOfVector(xk, xkn, n, iter);
                    if(iter>1) prevResult = result;
                    result = f.eval(xkn);
                    if (norm_vec < 1e-6)
                    {
                        break;
                    }
                    iter++;
                    for (int i = 0; i < n; i++)
                    {
                        xk[i] = xkn[i];
                    }
                    hessian = HessianMatrix(dff, xk);
                    // bool cholresult = checkCholesky(hessian);
                    grad = gradVector(df, xk);
                    for (int i = 0; i < n; i++)
                    {
                        grad[i] = -grad[i];
                    }
                    direcV = DirectionVector(hessian, grad);
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
                        hessian_modify_count+=1;
                        hessian = ModifyHessianOn2(func, fnc_param, xk, alp_options[i], hessian,dff);
                        // cholresult = checkCholesky(hessian);
                        for (int i = 0; i < n; i++)
                        {
                            grad[i] = -grad[i];
                        }
                        direcV = DirectionVector(hessian, grad);
                        for (int i = 0; i < n; i++)
                        {
                            grad[i] = -grad[i];
                        }
                        gfpk = 0.0000;
                        for (int i = 0; i < n; i++)
                        {
                            gfpk += direcV[i] * grad[i];
                        }
                        if(gfpk>0){
                            break;
                        }
                    }

                    for (int i = 0; i < n; i++)
                    {
                        xkn[i] = xk[i] + a * direcV[i];
                    }
                    IntervalVector func_value_xkn = f.eval(xkn);
                    while(func_value_xkn.lb()[0] > result.lb()[0] + c*a*gfpk){
                        a = a*rh;
                        if(a<a_min){
                            break;
                        }

                        for (int i = 0; i < n; i++)
                        {
                            xkn[i] = xk[i] + a * direcV[i];
                        }
                        func_value_xkn = f.eval(xkn);
                    }
                    if(a<a_min){
                        break;
                    }
                    a = a_init;
                    // fout << iter << "," << func_value_xkn.lb()[0] << "\n";
                    func_value_on2 = func_value_xkn.lb()[0];
                }
                fs_our = xkn;
                int on2_iter = iter;
                std :: cout << "Iterations: " << iter << " & Hessian Modified " << hessian_modify_count << " times." << endl;
                for (int i = 0; i < n; i++)
                {
                    std :: cout << fixed << setprecision(7) << xkn[i] << " ";
                }
                std :: cout << endl;

                // on3 method 


                a = a_init;
                iter = 1;
                xkn = inixk;
                hessian_modify_count = 0;
                cout << "ON3 Approach\n";
                while (true)
                {
                    if(iter>iter_max){
                        break;
                    }
                    double norm_vec = normOfVector(xk, xkn, n, iter);
                    if(iter>1) prevResult = result;
                    result = f.eval(xkn);
                    if (norm_vec < 1e-6)
                    {
                        break;
                    }
                    iter++;
                    for (int i = 0; i < n; i++)
                    {
                        xk[i] = xkn[i];
                    }
                    hessian = HessianMatrix(dff, xk);
                    // bool cholresult = checkCholesky(hessian);
                    grad = gradVector(df, xk);
                    for (int i = 0; i < n; i++)
                    {
                        grad[i] = -grad[i];
                    }
                    direcV = DirectionVector(hessian, grad);
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
                        hessian_modify_count+=1;
                        hessian = ModifyHessianOn3(func, fnc_param, xk, alp_options[i], hessian,dff);
                        // cholresult = checkCholesky(hessian);
                        for (int i = 0; i < n; i++)
                        {
                            grad[i] = -grad[i];
                        }
                        direcV = DirectionVector(hessian, grad);
                        for (int i = 0; i < n; i++)
                        {
                            grad[i] = -grad[i];
                        }
                        gfpk = 0.0000;
                        for (int i = 0; i < n; i++)
                        {
                            gfpk += direcV[i] * grad[i];
                        }
                        if(gfpk>0){
                            break;
                        }
                    }

                    for (int i = 0; i < n; i++)
                    {
                        xkn[i] = xk[i] + a * direcV[i];
                    }
                    IntervalVector func_value_xkn = f.eval(xkn);
                    while(func_value_xkn.lb()[0] > result.lb()[0] + c*a*gfpk){
                        a = a*rh;
                        if(a<a_min){
                            break;
                        }

                        for (int i = 0; i < n; i++)
                        {
                            xkn[i] = xk[i] + a * direcV[i];
                        }
                        func_value_xkn = f.eval(xkn);
                    }
                    if(a<a_min){
                        break;
                    }
                    a = a_init;
                    // fout << iter << "," << func_value_xkn.lb()[0] << "\n";
                    func_value_on3 = func_value_xkn.lb()[0];
                }
                fs_our = xkn;
                int on3_iter = iter;
                std :: cout << "Iterations: " << iter << " & Hessian Modified " << hessian_modify_count << " times." << endl;
                for (int i = 0; i < n; i++)
                {
                    std :: cout << fixed << setprecision(7) << xkn[i] << " ";
                }
                std :: cout << endl;

                fout << ger_iter << "," << on2_iter << "," << ds_iter << "," << on3_iter << "," << func_value_ger << "," << func_value_on2 << "," <<  func_value_ds << "," << func_value_on3 << "," << alp_options[i] << "\n";
            }
        }
        cout << "total " << total_test << " , better iterations: " << our_better << " , less value " << func_value_better << "times.\n";

        string dummy;
        getline(test_problems, dummy);
    }
    
    return 0;
}