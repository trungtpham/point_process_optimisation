% ------------------------------------------------------------------------ 
%  Copyright (C)
%  The Australian Center of Robotic Vision. The University of Adelaide
% 
%  Trung Pham <trung.pham@adelaide.edu.au>
%  April 2018
% ------------------------------------------------------------------------ 
% This file is part of the SceneCut method presented in:
%   T. T. Pham, H. Rezatofighi, T-J Chin, I. Reid 
%   Efficient Point Process Inference for Large-scale Object Detection 
%   CVPR 2016
% Please consider citing the paper if you use this code.

function [labels, energy] = lsa_tr_optimisation_tpham(unary_energy, pairwise_energy)

% This function implements the improved local submodularisation
% approximation algorithm for solving the nonsubmodular quadractic energy function:
% E(X) = U'*X + X'*V*X, where U is a real-valued, n-dimentional vector, 
% V is an n x n real symmetric matrix, X is a binary, n-dim vector.  
% Input: unary_energy (nx1): input unary energy vector, where N is the
% number of variables.
%        pairwise_energy (nxn) input pairwise energy matrix.
% Output: labels (nx1): output binary labels, one for each variable.
%         energy (scalar): optimised energy.


% Number of variables
n = length(unary_energy);

% Adjust pairwise energies based on unary energies
[rows, cols, v] = find(pairwise_energy);
alpha = abs(min([unary_energy(rows); unary_energy(cols)])) + 1e-3;
v = min(v, alpha');
pairwise_energy = full(sparse(rows, cols, v, n, n));

init_labels = 2.*ones(1,n);
[labels, energy] = lsa_tr_mex(unary_energy, pairwise_energy, init_labels);
end