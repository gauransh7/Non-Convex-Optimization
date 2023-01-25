#include <ibex.h>
#include <bits/stdc++.h>

using namespace ibex;
using namespace std;

int main() {
  // Define the function
  const ExprSymbol& x = ExprSymbol::new_("x", Dim::col_vec(3));
  Function f(x, x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);

  // Define the input point
  Vector point(3);
  point[0] = 1;
  point[1] = 2;
  point[2] = 7;

  // Calculate the value of the function for the input
  IntervalVector result = f.eval(point);

  // Print the result
  cout << "f(" << point << ") = " << result.lb()[0] << endl;

  return 0;
}
