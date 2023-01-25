#include <iostream>
#include <cmath>
#include <ibex.h>
#include <parser.h>

using namespace std;
using namespace ibex;

int main(int argc, char* argv[]) {
  // Parse the input arguments
  if (argc != 4) {
    cerr << "Usage: " << argv[0] << " <function> <gradient> <hessian>" << endl;
    return 1;
  }
  string function_str = argv[1];
  string gradient_str = argv[2];
  string hessian_str = argv[3];

  // Define the function, gradient, and Hessian
  const ExprSymbol& x = ExprSymbol::new_("x", Dim::col_vec(n));
  Function f(x, ExprParser::parse(function_str, x));
  Function grad_f(x, ExprParser::parse(gradient_str, x));
  Function hess_f(x, ExprParser::parse(hessian_str, x));

  // Initialize the starting point
  Vector x0(n);
  x0[0] = 0;
  x0[1] = 0;
  // Set the tolerance for the algorithm
  double tolerance = 1e-6;

  // Iterate until the tolerance is reached
  while (true) {
    // Calculate the search direction
    Vector p = -hess_f(x0).inverse() * grad_f(x0);

    // Check if the tolerance has been reached
    if (norm(p) < tolerance) {
      break;
    }

    // Update the point
    x0 = x0 + p;
  }

  // Print the result
  cout << "Minimum found at x = " << x0 << endl;

  return 0;
}