/* in this code we are using c++ for optimization
   we are using fadbad pakage for automatic differentiation
   using fadbad we have calculated gradient vector and hessian matrix
   we uses another pakage Eigen for solving linear algebra (matrix equation
*/


#include <bits/stdc++.h>
#include "tadiff.h"
#include "fadbad.h"
#include "fadiff.h"
#include "badiff.h"
#include"eigen/Eigen/Dense"
#include "eigen/Eigen/QR"




using namespace std;
using namespace fadbad;
//this function convert std c++ vector to Eigen vector
Eigen::VectorXd ConvertToEigenVector(vector<double> v){
    double* ptr_data = &v[0];
    Eigen::VectorXd v2 = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(v.data(), v.size());
    return v2;
}
//this function convert Eigen vector to std c++ vector
vector<double>ConvertTo1dVector(Eigen::VectorXd mat){
    vector<double> vec(mat.data(), mat.data() + mat.rows() * mat.cols());
    return vec;
}

//this function convert Eigen Matrix to std c++ 2d vector
vector<vector<double>> ConvertTo2dVector(Eigen::MatrixXd m){
  
std::vector<std::vector<double>> v;

for (int i=0; i<m.rows(); ++i)
{
    const double* begin = &m.row(i).data()[0];
    v.push_back(std::vector<double>(begin, begin+m.cols()));
}

return v;
}

//this function convert std c++ 2d vector to Eigen Matrix
Eigen::MatrixXd ConvertToEigenMatrix(std::vector<std::vector<double>> data)
{
    Eigen::MatrixXd eMatrix(data.size(), data[0].size());
    for (int i = 0; i < data.size(); ++i)
        eMatrix.row(i) = Eigen::VectorXd::Map(&data[i][0], data[0].size());
    return eMatrix;
}


//this function return our function to which want to optimize
//if we need to optimise another function than we have to change inside this function
B< F<double> > func( B< F<double> > x[], int n)
{
	// B<F<double>>z;
    // for(int i=0;i<n;i++){
    //     z+=(x[i]*cos(x[i]));
    // }

	// return (3*x[0]*x[0]+2*x[1]*x[1]+x[2]*x[2]*x[2]-2*x[0]*x[1]-2*x[0]*x[2]+2*x[1]*x[2]-6*x[0]-4*x[1]-2*x[2]);
    // return (2*x[0]*x[0]*x[0]*x[1]*x[1]*x[1]*x[1]*x[2]*x[3]*x[3]-2*x[1]*x[1]*x[1]*x[3]*x[3]*x[3]*x[3]+3*x[0]*x[0]*x[2]*x[2]);
    // return (x[0]*x[0]+x[1]*x[1]+3*x[2]*x[2]+4*x[3]+6*x[4]);
    // return (x[0]*x[0]*x[0]*x[0]*x[0]+x[1]*x[1]*x[1]*x[1]+x[2]*x[2]*x[2]+x[3]*x[3]+x[4]);
    return (x[0]*x[0]+2*x[1]*x[1]+3*x[2]*x[2]+2*x[1]*x[0]+2*x[1]*x[2]);
}

// this function return a gradient vector for our function
vector<double>gradVector(vector<double>arr,int n){
    vector<double>g(n);
    B<F<double>>x[n],f;
    for(int i=0;i<n;i++){
        x[i]=arr[i];
    }
    for(int i=0;i<n;i++){
        x[i].x().diff(i,n);
    }
    f=func(x,n);
     f.diff(0,1);
    
    double fval=f.x().x();
    for(int i=0;i<n;i++){
       
g[i]=x[i].d(0).x();
         }


         return g;
}

//this function return norm of vector (L2)
double normOfVector(vector<double>g,int n){
    double gr;
    for(int i=0;i<n;i++){
        gr+=(g[i]*g[i]);
    }
    return sqrt(gr);
}

//this function return hessian matrix
vector<vector<double>> hmat(vector<double> arr,int n){
    vector<vector<double>> h1(n,vector<double>(n));
    B<F<double>>x[n],f;
    for(int i=0;i<n;i++){
        x[i]=arr[i];
    }
    for(int i=0;i<n;i++){
        x[i].x().diff(i,n);
    }
    
    // double fval=f.x().x();
    vector<double>g1(n);
    for(int i=0;i<n;i++){
g1[i]=x[i].d(0).x();
         }
f=func(x,n);
      f.diff(0,1);
 for(int i=0;i<n;i++){
     
        for(int j=0;j<n;j++){
            
   
            h1[i][j]=x[j].d(0).d(i);
        }
    }
    // h1[n-1][n-1]=x[n-1].d(n-1).d(n-1);
    return h1;

}

//this function return direction vector 
//we solve linear algebra equation  for x:  Ax=b,
// x is our directional vector Pk
vector<double>DirectionVector(vector<vector<double>>v,vector<double>B){
    int n=B.size();
    Eigen::MatrixXd A(n,n);
A=ConvertToEigenMatrix(v);
Eigen::MatrixXd b(n,1);
b=ConvertToEigenVector(B);
Eigen::VectorXd  x = A.colPivHouseholderQr().solve(b);
vector<double>X(n);
    X=ConvertTo1dVector(x);
   
return X;
}
int main()
{cout<<"input n:";
    int n;
    cin>>n;
    //vector arr is our initial point which we give input
    //vector g is gradient vector
vector<double> arr(n),g(n);

//2d vector h is hessian matrix
vector<vector<double>> h(n,vector<double>(n));


	 for(int i=0;i<n;i++){
         cin>>arr[i];
        
     }  

     // vector p is direction vector Pk                    
vector<double>p(n);

//alpha intial value is 1
double a=1.0000;

//c constant used for check strong wolf condition
double c=0.9000;

//rho value to update alpha value 
double rh=0.9;



while(normOfVector(arr,n)>=1e-6){
h=hmat(arr,n);
g=gradVector(arr,n);
for(int i=0;i<n;i++){
    g[i]=-g[i];
}
p=DirectionVector(h,g);
for(int i=0;i<n;i++){
    g[i]=-g[i];
}
double gfpk=0.0000;
for(int i=0;i<n;i++){
    gfpk+=p[i]*g[i];
}
vector<double>temp(n);

for(int i=0;i<n;i++){
    temp[i]=arr[i]+a*p[i];
}
double gfpk1=0.0000;
for(int i=0;i<n;i++){
    gfpk1+=(temp[i]*p[i]);
}
while(!(abs(gfpk1)<=(c*abs(gfpk)))){
    a=a*rh;
    for(int i=0;i<n;i++){
    temp[i]=arr[i]+a*p[i];
}
gfpk1=0.0000;
for(int i=0;i<n;i++){
    gfpk1+=(temp[i]*p[i]);
}


}
for(int i=0;i<n;i++){
    arr[i]=temp[i];
}



}

//this print our final x*(point of minima)
for(int i=0;i<n;i++){
   
    cout<<fixed<<setprecision(7)<<arr[i]<<" ";
}cout<<endl;
	return 0;
}
