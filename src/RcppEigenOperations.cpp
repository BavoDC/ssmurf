// [[Rcpp::depends(RcppArmadillo, RcppEigen)]]
#define ARMA_64BIT_WORD 1
#include <RcppEigen.h>

using namespace Rcpp;
using namespace Eigen;
using Eigen::Lower;
using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::Upper;
using Eigen::VectorXd;
using Eigen::VectorXf;

// [[Rcpp::export]]
Eigen::VectorXd XtV(Eigen::SparseMatrix<double> X, const Eigen::Map<Eigen::VectorXd> V){
  Eigen::VectorXd C = X.transpose() * V;
  return C;
}

// [[Rcpp::export]]
Eigen::VectorXd XB(Eigen::SparseMatrix<double> X, const Eigen::Map<Eigen::VectorXd> B){
  Eigen::VectorXd C = X * B;
  return C;
}

// [[Rcpp::export]]
Eigen::SparseMatrix<double> XXt(Eigen::SparseMatrix<double> X, Eigen::SparseMatrix<double> Xt){
  Eigen::SparseMatrix<double> C = X * Xt;
  return C;
}