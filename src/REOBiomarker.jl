module REOBiomarker

using Random, Statistics, StatsBase, Distributions
using DataFrames, Lasso, GLM, Combinatorics, ThreadsX
using DecisionTree, HypothesisTests, JLD2
using StatisticalMeasures, CategoricalArrays, CategoricalDistributions

export REOConfig, REOModel
export fit_reo, predict_reo, evaluate_reo, run_permutation_test, generate_test_data

# Traditional TSP baselines
export TSPModel, KTSPModel, AUCTSPModel
export fit_tsp, predict_tsp, fit_ktsp, predict_ktsp, fit_auctsp, predict_auctsp
export evaluate_tsp, evaluate_ktsp, evaluate_auctsp

include("types.jl")       # Type definitions
include("filters.jl")     # Gene and gene-pair filtering
include("training.jl")    # Stability selection and model fitting
include("vote.jl")        # Majority voting feature subset search
include("evaluation.jl")  # Prediction, evaluation, and permutation tests
include("utils.jl")       # Test data generation utilities
include("statistics.jl")  # Bayesian quality control and tau/alpha estimation
include("tsp.jl")         # Classical Top Scoring Pair
include("ktsp.jl")        # Classical k-Top Scoring Pairs
include("auctsp.jl")      # Classical AUC-based Top Scoring Pairs

end
