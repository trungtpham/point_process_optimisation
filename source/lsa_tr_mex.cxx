/*
 ------------------------------------------------------------------------ 
  Copyright (C)
  The Australian Center of Robotic Vision. The University of Adelaide
 
  Trung Pham <trung.pham@adelaide.edu.au>
  April 2018
 ------------------------------------------------------------------------ 
 This file is part of the SceneCut method presented in:
   T. T. Pham, H. Rezatofighi, T-J Chin, I. Reid 
   Efficient Point Process Inference for Large-scale Object Detection 
   CVPR 2016
 Please consider citing the paper if you use this code.
*/

/* LSA TR C++ implementation */
// Eigen
#include <eigen3/Eigen/Dense>

#include <stdlib.h>
#include <cmath>
#include <limits.h>
#include <time.h>
#include <algorithm>
#include <boost/format.hpp>

// Boost
#include <boost/config.hpp>
#include <iostream>
#include <vector>
#include <utility>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

//typedef uint16_t char16_t; // "unknown type name 'char16_t'" under Mac OS 10.9
#include "mex.h"
#include <ctime>

double computeEnergy(const Eigen::VectorXd &unary, const Eigen::MatrixXd &pairwise, const Eigen::VectorXd &labelling);
void computeApproxUnaryTerms(uint32_t size_n, Eigen::VectorXd* approxUnary, const Eigen::VectorXd &unary, const Eigen::MatrixXd &pairwise, const Eigen::VectorXd &currLabeling);
double computeApproxEnergy(const Eigen::VectorXd &approxUnary, const Eigen::VectorXd &labeling);
void computeApproxLabeling(uint32_t size_n, Eigen::VectorXd* lambdaLabeling, double lambda, const Eigen::VectorXd &approxUnary, const Eigen::VectorXd &currLabeling);
void findMinimalChangeBreakPoint(uint32_t size_n, double* bestLambda, Eigen::VectorXd* bestLabeling, const Eigen::VectorXd &approxUnary, const Eigen::VectorXd &currLabeling, double currlambda);
void LSA_TR(double* outputEnergy, Eigen::VectorXd* outputLabeling, uint32_t size_n, const Eigen::VectorXd &unary, const Eigen::MatrixXd &pairwise, const Eigen::VectorXd &initLabeling);


const double LAMBDA_LAGRANGIAN = 0.1;   
const double REDUCTION_RATIO_THRESHOLD = 0.25;
const double MAX_LAMBDA_LAGRANGIAN = 1e5;
const double LAMBDA_MULTIPLIER = 1.5;
const double PRECISION_COMPARE_GEO_LAMBDA = 1e-9;
const double LAMBDA_LAGRANGIAN_RESTART = 0.1;

void mexFunction(int nargout, mxArray *out[], int nargin, const mxArray *in[]){

	uint32_t size_n;
    double* unary;
	double* pairwise;
	double* initLabeling;
	double* outputLabeling;
	double* outputEnergy;
    
    
	/* Get input */
	size_n = std::max((int)mxGetN(in[0]), (int)mxGetM(in[0]));
	unary = mxGetPr(in[0]);
	pairwise = mxGetPr(in[1]);
	initLabeling = mxGetPr(in[2]);

	/* Create outputs */
	out[0] = mxCreateDoubleMatrix(size_n,1,mxREAL);
	outputLabeling = mxGetPr(out[0]);
	out[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
	outputEnergy = mxGetPr(out[1]);

        Eigen::VectorXd eInitLabeling = Eigen::Map<Eigen::MatrixXd> (initLabeling, size_n, 1);
	Eigen::VectorXd eUnary = Eigen::Map<Eigen::MatrixXd> (unary, size_n, 1);
        Eigen::MatrixXd ePairwise = Eigen::Map<Eigen::MatrixXd> (pairwise, size_n, size_n); 

        Eigen::VectorXd finalLabeling(size_n);
	double finalEnergy;
	clock_t begin = std::clock();
        LSA_TR(&finalEnergy, &finalLabeling, size_n, eUnary, ePairwise, eInitLabeling);
	clock_t end = std::clock();
	std::cout << "Time (s) " << double(end - begin) / CLOCKS_PER_SEC << "\n";
	Eigen::Map<Eigen::VectorXd>(outputLabeling, size_n) = finalLabeling;
	*outputEnergy = finalEnergy;
}

// unary is a nx2 matrix, pairwise is a nxn matrix
void LSA_TR(double* outputEnergy, Eigen::VectorXd* outputLabeling, uint32_t size_n, 
	    const Eigen::VectorXd &unary, const Eigen::MatrixXd &pairwise, const Eigen::VectorXd &initLabeling){

	Eigen::VectorXd currLabeling;
	currLabeling = initLabeling;
	double currEnergy = computeEnergy(unary, pairwise, currLabeling);
	double lambdaEnergy = 0;
	Eigen::VectorXd approxUnary(size_n);
	computeApproxUnaryTerms(size_n, &approxUnary, unary, pairwise, currLabeling);
	double lambda = LAMBDA_LAGRANGIAN;
	double actualReduction = 0;
	double predictedReduction = 0;
        bool stopFlag = false;
 	while (stopFlag==false){

                //Compute lamda Labelling
		Eigen::VectorXd lambdaLabeling;
		computeApproxLabeling(size_n, &lambdaLabeling, lambda, approxUnary, currLabeling);
                double currApproxE = computeApproxEnergy(approxUnary, currLabeling);
                double lambdaApproxE = computeApproxEnergy(approxUnary, lambdaLabeling);
		predictedReduction = currApproxE - lambdaApproxE;
		if (predictedReduction < 0) {std::cout << "Negative reduction \n"; stopFlag = true;}
		bool updateSolutionFlag = false;
		// there is no updates, find another breaking point (ie smaller lambda)
		if (lambdaLabeling == currLabeling){
			//std::cout << "There is no updates, find a better lambda using linear search \n";
			findMinimalChangeBreakPoint(size_n, &lambda, &lambdaLabeling, approxUnary, currLabeling, lambda);
			//std::cout << "new lambda " << lambda << "\n";
			lambdaApproxE = computeApproxEnergy(approxUnary, lambdaLabeling);
			lambdaEnergy = computeEnergy(unary, pairwise, lambdaLabeling);
			predictedReduction = currApproxE - lambdaApproxE;
			if (predictedReduction < 0){
				std::cout << "Negative predicted reduction \n";
				stopFlag = true;
			}
			actualReduction = currEnergy - lambdaEnergy;
			if (actualReduction <= 0 || lambdaLabeling.sum()== 0 || lambdaLabeling.sum()== size_n){
				stopFlag = true;
				std::cout << "Optimization done! \n";
			}
			else{
				if (lambda == 0) lambda = LAMBDA_LAGRANGIAN_RESTART;
				updateSolutionFlag = true;
			}	

		}else{
			// Compute actual energy with lambda labeling
			lambdaEnergy = computeEnergy(unary, pairwise, lambdaLabeling);
			actualReduction = currEnergy - lambdaEnergy;
			if (actualReduction <= 0) updateSolutionFlag = false;
			else updateSolutionFlag = true;
		}

		// If we don't stop, update solution, re-adjust lamdba parameter.
		if (stopFlag == false){
			double reductionRatio = actualReduction/predictedReduction;
			if (reductionRatio < REDUCTION_RATIO_THRESHOLD){
				if (lambda < MAX_LAMBDA_LAGRANGIAN)
					lambda *= LAMBDA_MULTIPLIER;
			}
			else{
				if (lambda > PRECISION_COMPARE_GEO_LAMBDA)
					lambda /= LAMBDA_MULTIPLIER;
			}
  			// Update solution
			if (updateSolutionFlag == true){
				currLabeling = lambdaLabeling;
				currEnergy =  lambdaEnergy;
				computeApproxUnaryTerms(size_n, &approxUnary, unary, pairwise, currLabeling);
			}
			
		}
	}
	*outputEnergy = currEnergy;
	*outputLabeling = currLabeling;
}

double computeEnergy(const Eigen::VectorXd &unary, const Eigen::MatrixXd &pairwise, const Eigen::VectorXd &labeling){

	double UE = unary.dot(labeling);
        double PE = 0;
        Eigen::VectorXd  temp = labeling.transpose()*pairwise;
        PE = temp.dot(labeling);
        return UE + PE;
}

double computeApproxEnergy(const Eigen::VectorXd &approxUnary, const Eigen::VectorXd &labeling){

	double E = approxUnary.dot(labeling);
        return E;
}

void computeApproxLabeling(uint32_t size_n, Eigen::VectorXd* lambdaLabeling, double lambda,
			  const Eigen::VectorXd &approxUnary, const Eigen::VectorXd &currLabeling){

	// Hamming distance from the current labelling
	Eigen::MatrixXd distUE(2,size_n);
	distUE = Eigen::MatrixXd::Zero(2,size_n);
	distUE.row(0) = currLabeling;
	distUE.row(1) = Eigen::VectorXd::Ones(size_n) - currLabeling;

	Eigen::MatrixXd approxUnaryAll(2, size_n);
	approxUnaryAll.row(0) = lambda*distUE.row(0);
	approxUnaryAll.row(1) = approxUnary.transpose() + lambda*distUE.row(1);

	Eigen::VectorXd temp = approxUnaryAll.row(0) - approxUnaryAll.row(1);
	*lambdaLabeling = (temp.array() < 0).select(Eigen::VectorXd::Zero(size_n),Eigen::VectorXd::Ones(size_n));


}

void computeApproxUnaryTerms(uint32_t size_n, Eigen::VectorXd* approxUnary,
			     const Eigen::VectorXd &unary, const Eigen::MatrixXd &pairwise, const Eigen::VectorXd &currLabeling){
        Eigen::VectorXd approxPairwise = currLabeling.transpose()*pairwise;
	*approxUnary = unary.transpose() + 2*approxPairwise.transpose();
}


void findMinimalChangeBreakPoint(uint32_t size_n, double* bestLambda, Eigen::VectorXd* bestLabeling,
				 const Eigen::VectorXd &approxUnary, const Eigen::VectorXd &currLabeling, double currlambda){

	bool foundLambda = false;

	// Hamming distance from the current labelling
	Eigen::MatrixXd distUE(2,size_n);
	distUE = Eigen::MatrixXd::Zero(2,size_n);
	distUE.row(0) = currLabeling;
	distUE.row(1) = Eigen::VectorXd::Ones(size_n) - currLabeling;

	double topLambda = currlambda;
	Eigen::VectorXd topLabeling;
	computeApproxLabeling(size_n, &topLabeling, topLambda, approxUnary, currLabeling);


	while ((topLabeling == currLabeling) == false){
		topLambda *= LAMBDA_MULTIPLIER;
		computeApproxLabeling(size_n, &topLabeling, topLambda, approxUnary, currLabeling);
	}

	double bottomLambda = PRECISION_COMPARE_GEO_LAMBDA;
	Eigen::VectorXd bottomLabeling;
	computeApproxLabeling(size_n, &bottomLabeling, bottomLambda, approxUnary, currLabeling);

	while (foundLambda==false){
		double middleLambda = 0.5*topLambda + 0.5*bottomLambda;
		
		Eigen::VectorXd middleLabeling;
		computeApproxLabeling(size_n, &middleLabeling, middleLambda, approxUnary, currLabeling);
		if ((middleLabeling == topLabeling) == false){
			bottomLambda = middleLambda;
			bottomLabeling = middleLabeling;
		}else if ((middleLabeling == bottomLabeling) == false){
			topLambda = middleLambda;
			topLabeling = middleLabeling;
		}else{
			foundLambda = true;
		}
		if ((topLambda - bottomLambda) < PRECISION_COMPARE_GEO_LAMBDA)	foundLambda = true;	
	}
	*bestLambda = bottomLambda;
	*bestLabeling = bottomLabeling;
}





