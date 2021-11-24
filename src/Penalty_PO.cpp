// [[Rcpp::depends(RcppArmadillo, RcppEigen)]]
#include <RcppArmadillo.h>
#include <RcppEigen.h>
using namespace Rcpp;
using namespace arma;
using namespace Eigen;
using Eigen::Lower;
using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::Upper;
using Eigen::VectorXd;
using Eigen::VectorXf;


// Soft thresholding operator S_y(x) where x is a vector
Eigen::VectorXd soft_thresh(const Eigen::VectorXd x, const double y) {
  Eigen::VectorXd z = Eigen::VectorXd::Zero(x.size());
  for (unsigned int i = 0; i < x.size(); i++) {
    //z(i) = std::max(0.0, x(i) - y) - std::max(0.0, -x(i) - y);
    // sign(x) * max(|x| - y, 0)
    z(i) = std::copysign(1, x(i)) * std::max(std::abs(x(i)) - y, 0.0);
  }
  return z;
}

// Soft thresholding operator S_y(x) where both x and y are vectors
Eigen::VectorXd soft_thresh_vec(const Eigen::VectorXd x, const Eigen::VectorXd y) {
  Eigen::VectorXd z = Eigen::VectorXd::Zero(x.size());
  for (unsigned int i = 0; i < x.size(); i++) {
    // sign(x) * max(|x| - y, 0)
    z(i) = std::copysign(1, x(i)) * std::max(std::abs(x(i)) - y(i), 0.0);
  }
  return z;
}

// Group soft thresholding operator S_y(x) where x is a vector
Eigen::VectorXd group_soft_thresh(const Eigen::VectorXd x, const double y) {
  double norm_x = x.norm();
  Eigen::VectorXd z = Eigen::VectorXd::Zero(x.size());
  
  // Avoid problems if negative or zero norm
  if (norm_x > 0) {
    for (unsigned int i = 0; i < x.size(); i++) {
      z(i) = x(i) * std::max(1 - y / norm_x, 0.0);
    }
  }
  
  return z;
}


// [[Rcpp::export]]
Eigen::VectorXd admm_po_cpp(const Eigen::Map<Eigen::VectorXd>& beta_tilde, const double slambda, const Eigen::Map<Eigen::VectorXd>& lambda1, const double lambda2, 
                          const Eigen::Map<Eigen::MatrixXd>& penmat, const Eigen::Map<Eigen::MatrixXd>& Q, 
                          const Eigen::Map<Eigen::VectorXd>& eigval, const bool fast,
                          const int maxiter, double rho, const Eigen::Map<Eigen::VectorXd>& beta_old) {
  
  Eigen::MatrixXd penmat_t = penmat.transpose();
  int m = penmat.rows();
  int d = penmat.cols();
  
  // Initialize values
  Eigen::VectorXd x = Eigen::VectorXd::Zero(d);
  Eigen::VectorXd xhat = Eigen::VectorXd::Zero(d);
  Eigen::VectorXd z_old = Eigen::VectorXd::Zero(m);
  // Use starting value for beta
  Eigen::VectorXd z_new = penmat * beta_old;
  // Starting value for u is zero Eigen::VectorXdtor
  Eigen::VectorXd u = Eigen::VectorXd::Zero(m);
  
  // Relative tolerance
  double eps_rel = pow(10, -10.0);
  // Absolute tolerance
  double eps_abs = pow(10, -12.0);
  // Tolerance for primal feasibility condition
  double eps_pri = sqrt((double) m) * eps_abs + eps_rel * std::max((penmat * x).norm(), z_new.norm());
  // Tolerance for dual feasibility condition
  double eps_dual = sqrt((double) d) * eps_abs + eps_rel * rho * (penmat_t * u).norm();
  // Norm of primal residuals
  double r_norm = (penmat * x - z_new).norm();
  // Norm of dual residuals
  double s_norm = (- rho * penmat_t * (z_new - z_old)).norm();
  
  
  // Relaxation parameter
  double xi = 1.5;
  // Iteration counter, note that we start from 1!
  int iter = 1;
  
  Eigen::MatrixXd ADMM_aux;
  Eigen::MatrixXd Qt;
  Eigen::MatrixXd D = Eigen::VectorXd::Ones(d).asDiagonal();
  // Auxiliary matrix
  if (fast) {
    Qt = Q.transpose();
    Eigen::MatrixXd Mt = Eigen::VectorXd (1/(eigval.array().pow(-1) + rho)).asDiagonal();
    // Fast version to compute inverse
    ADMM_aux = D - rho * Q * Mt * Qt;
  } else {
    // Slower (standard) version to compute inverse
    ADMM_aux = (D + rho * penmat_t * penmat).inverse();
  }
  
  
  // Parameters to update rho, see Zhu (2017), JCGS
  double mu = 10;
  double eta = 2;
  double rho_old;
  //Eigen::VectorXd G(d);
  
  
  while ((r_norm > eps_pri || s_norm > eps_dual || iter==1) && iter < maxiter) {
    
    // Check for interrupt every 1000 iterations
    if (iter % 1000 == 0) {
      Rcpp::checkUserInterrupt();
    }
    
    
    z_old = z_new;
    //G = ((z_old - u).t() * penmat).t();
    
    // Update x
    //x = ADMM_aux * (beta_tilde + rho * G);
    x = ADMM_aux * (beta_tilde + rho * penmat_t * (z_old - u));
    
    // Relaxation
    xhat = xi * penmat * x + (1 - xi) * z_old;
    // Update z
    z_new = soft_thresh(xhat + u, slambda / rho);
    // Update u
    u = u + xhat - z_new;
    
    
    // Tolerance for primal feasibility condition
    eps_pri = sqrt((double) m) * eps_abs + eps_rel * std::max((penmat * x).norm(), z_new.norm());
    // Tolerance for dual feasibility condition
    eps_dual = sqrt((double) d) * eps_abs + eps_rel * rho * (penmat_t * u).norm();
    // Norm of primal residuals
    r_norm = (penmat * x - z_new).norm();
    // Norm of dual residuals
    s_norm = (- rho * penmat_t * (z_new - z_old)).norm();
    
    
    // Check if rho needs to be increased
    if (r_norm / eps_pri >= mu * s_norm / eps_dual) {
      
      rho_old = rho;
      // Update rho
      rho = eta * rho_old;
      Eigen::MatrixXd D = Eigen::VectorXd::Ones(d).asDiagonal();
      // Update auxiliary matrix
      if (fast) {
        // Fast version to compute inverse
        Eigen::MatrixXd Mt = Eigen::VectorXd (1/(eigval.array().pow(-1) + rho)).asDiagonal();
        // Fast version to compute inverse
        ADMM_aux = D - rho * Q * Mt * Qt;
        //ADMM_aux = eye(d, d) - rho * Q * diagmat(1/(1/eigval + rho)) * Qt;
        
      } else {
        // Slower (standard) version to compute inverse
        ADMM_aux = (D + rho * penmat_t * penmat).inverse();
      }
      // update u
      u = u * rho_old / rho;
      
      // Check if rho needs to be decreased
    } else if (s_norm / eps_dual >= mu * r_norm / eps_pri) {
      
      rho_old = rho;
      // Update rho
      rho = rho_old / eta;
      Eigen::MatrixXd D = Eigen::VectorXd::Ones(d).asDiagonal();
      // Update auxiliary matrix
      if (fast) {
        // Fast version to compute inverse
        Eigen::MatrixXd Mt = VectorXd (1/(eigval.array().pow(-1) + rho)).asDiagonal();
        ADMM_aux = D - rho * Q * Mt * Qt;        
      } else {
        // Slower (standard) version to compute inverse
        ADMM_aux = (D + rho * penmat_t * penmat).inverse();
      }
      // update u
      u = u * rho_old / rho;
    }
    
    iter++;
  }
  
  if (lambda2 > 0) {
    // In case lambda2 is non-zero, an extra step is needed to obtain the proximal operator (see Liu et al. (2010))
    x = group_soft_thresh(x, slambda * lambda2);
  }
  
  // In case lambda1 is non-zero, an extra step is needed to obtain the proximal operator (see Liu et al. (2010))
  if (lambda1.size() > 1) {
    // Vector version
    x = soft_thresh_vec(x, slambda * lambda1);
    
  } else {
    // Only one element, check if non-zero first
    if (lambda1(0) > 0) {
      x = soft_thresh(x, slambda * lambda1(0));
    }
  }
  
  return x;
}


