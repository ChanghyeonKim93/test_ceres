#include <iostream>
#include <ceres/ceres.h>

struct CostFunctor {
   template <typename T>
   bool operator()(const T* const x, T* residual) const {
     residual[0] = 10.0 - x[0];
     return true;
   }
};

class AnalyicCostFunction : public ceres::SizedCostFunction<1, 1> {
 public:
  virtual ~AnalyicCostFunction() {}
  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    const double x = parameters[0][0];
    residuals[0] = 10 - x*x;

    // Compute the Jacobian if asked for.
    if (jacobians != nullptr && jacobians[0] != nullptr) {
      jacobians[0][0] = -2.0*x;
    }
    return true;
  }
};

int main()
{
  // The variable to solve for with its initial value.
  double initial_x = 522.0;
  double x = initial_x;

  // Build the problem.
  ceres::Problem problem;

  // Set up the only cost function (also known as residual). This uses
  // auto-differentiation to obtain the derivative (jacobian).
  // ceres::CostFunction* cost_function =
  //     new ceres::AutoDiffCostFunction<CostFunctor, 1, 1>(new CostFunctor);
  
  ceres::CostFunction* cost_function = new AnalyicCostFunction;

  problem.AddResidualBlock(cost_function, nullptr, &x);

  // Run the solver!
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  std::cout << summary.BriefReport() << "\n";
  std::cout << "x : " << initial_x
            << " -> " << x << "\n";
  return 1;
};