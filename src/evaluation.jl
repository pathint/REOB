# src/evaluation.jl

using Statistics, StatisticalMeasures, CategoricalArrays, CategoricalDistributions

"""
    predict_reo(model, test_data, test_gene_ids) -> (probs, preds)

Predict class probabilities and binary labels for new samples.

For `LassoMethod` the score is passed through a sigmoid; for `RFMethod` and
`VotingMethod` the weighted vote is clamped to [0, 1].  Predictions use a
fixed threshold of 0.5.
"""
function predict_reo(model::REOModel, test_data::Matrix{Float64}, test_gene_ids::Vector)
    n_samples = size(test_data, 2)
    gene_to_idx = Dict(name => i for (i, name) in enumerate(test_gene_ids))

    # Filter to gene pairs present in the test set
    available_indices = Int[]
    available_weights = Float64[]
    active_pairs = []

    for (i, pair) in enumerate(model.final_pairs)
        g1_name, g2_name = pair
        if haskey(gene_to_idx, g1_name) && haskey(gene_to_idx, g2_name)
            push!(active_pairs, (gene_to_idx[g1_name], gene_to_idx[g2_name]))
            push!(available_weights, model.weights[i])
            push!(available_indices, i)
        end
    end

    isempty(active_pairs) && error("No model gene pairs found in the test set.")

    # Build binary feature matrix
    X_val = zeros(Float64, n_samples, length(active_pairs))
    for (j, (g1_idx, g2_idx)) in enumerate(active_pairs)
        @views X_val[:, j] .= test_data[g1_idx, :] .> test_data[g2_idx, :]
    end

    z = (X_val * available_weights) .+ model.intercept

    if model.config.method == LassoMethod
        probs = 1.0 ./ (1.0 .+ exp.(-z))
    else
        probs = clamp.(z, 0, 1)
    end
    preds = probs .>= 0.5
    return (probs=probs, preds=preds)
end

"""
    evaluate_reo(model, valid_data, valid_gene_ids, valid_labels) -> NamedTuple

Compute accuracy, MCC, AUC, and confusion matrix for a trained REO model
on a validation dataset.
"""
function evaluate_reo(
    model::REOModel,
    valid_data::Matrix{Float64},
    valid_gene_ids::Vector,
    valid_labels::AbstractVector,
)
    res = predict_reo(model, valid_data, valid_gene_ids)

    acc = mean(res.preds .== valid_labels)
    mcc = _calculate_mcc(res.preds, valid_labels)

    y_true = categorical(valid_labels; ordered=true)
    y_pred = UnivariateFinite([0, 1], res.probs; augment=true, pool=y_true)
    auc_val = auc(y_pred, y_true)

    tp = sum((res.preds .== 1) .& (valid_labels .== 1))
    tn = sum((res.preds .== 0) .& (valid_labels .== 0))
    fp = sum((res.preds .== 1) .& (valid_labels .== 0))
    fn = sum((res.preds .== 0) .& (valid_labels .== 1))

    if model.config.verbose
        println("--- REO evaluation ($(model.config.method)) ---")
        println("  Pairs:  $(length(model.final_pairs))")
        println("  ACC:    $(round(acc, digits=4))")
        println("  MCC:    $(round(mcc, digits=4))")
        println("  AUC:    $(round(auc_val, digits=4))")
        println("  Confusion: [TP=$tp, FP=$fp; FN=$fn, TN=$tn]")
    end

    return (acc=acc, mcc=mcc, auc=auc_val, probs=res.probs, preds=res.preds)
end

"""
    _calculate_mcc(preds, labels) -> Float64

Compute the Matthews Correlation Coefficient.
"""
function _calculate_mcc(preds::BitVector, labels::AbstractVector)
    tp = Float64(sum(preds .& (labels .== 1)))
    tn = Float64(sum(.!preds .& (labels .== 0)))
    fp = Float64(sum(preds .& (labels .== 0)))
    fn = Float64(sum(.!preds .& (labels .== 1)))

    num = (tp * tn) - (fp * fn)
    den = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return den == 0 ? 0.0 : num / den
end

function _calculate_mcc(preds::Vector{Bool}, labels::AbstractVector)
    _calculate_mcc(BitVector(preds), labels)
end

"""
    run_permutation_test(data, labels, genes, model, cfg; n_permutations=100)

Full permutation test: refits the model on each permuted label set.
"""
function run_permutation_test(
    data, labels, genes, model::REOModel, cfg::REOConfig; n_permutations=100
)
    cfg.verbose && println(">>> Running permutation test ($n_permutations permutations)...")

    original_model = fit_reo(data, labels, genes, cfg)
    res = evaluate_reo(model, data, genes, labels)
    observed_mcc = res.mcc

    permuted_mccs = zeros(n_permutations)
    for i in 1:n_permutations
        shuffled_labels = shuffle(labels)
        try
            p_model = fit_reo(data, shuffled_labels, genes, cfg)
            p_res = evaluate_reo(p_model, data, genes, shuffled_labels)
            permuted_mccs[i] = p_res.mcc
        catch
            permuted_mccs[i] = 0.0
        end
    end

    p_value = sum(permuted_mccs .>= observed_mcc) / n_permutations
    return (p_value=p_value, observed_mcc=observed_mcc, permuted_mccs=permuted_mccs)
end


"""
    evaluate_tsp(model, valid_data, valid_gene_ids, valid_labels; verbose = false)

Return accuracy, MCC, and prediction result for a trained TSP model
on a validation dataset.
"""
function evaluate_tsp(
    model::TSPModel,
    valid_data::Matrix{Float64},
    valid_gene_ids::Vector,
    valid_labels::AbstractVector;
	verbose::Bool = false
)
    res = predict_tsp(model, valid_data, valid_gene_ids)

    acc = mean(res .== valid_labels)
    mcc = _calculate_mcc(res, valid_labels)

    tp = sum((res .== 1) .& (valid_labels .== 1))
    tn = sum((res .== 0) .& (valid_labels .== 0))
    fp = sum((res .== 1) .& (valid_labels .== 0))
    fn = sum((res .== 0) .& (valid_labels .== 1))

    if verbose 
       println("--- TSP evaluation ---")
       println("  ACC: $(round(acc, digits=4))")
       println("  MCC: $(round(mcc, digits=4))")
       println("  Confusion: [TP=$tp, FP=$fp; FN=$fn, TN=$tn]")
	end

    return (acc=acc, mcc=mcc, preds=res)
end


"""
    evaluate_ktsp(model, valid_data, valid_gene_ids, valid_labels; verbose = false)

Return accuracy, MCC, and prediction result for a trained KTSP model
on a validation dataset.
"""
function evaluate_ktsp(
    model::KTSPModel,
    valid_data::Matrix{Float64},
    valid_gene_ids::Vector,
    valid_labels::AbstractVector;
	verbose::Bool = false
)
    res = predict_ktsp(model, valid_data, valid_gene_ids)

    acc = mean(res .== valid_labels)
    mcc = _calculate_mcc(res, valid_labels)

    tp = sum((res .== 1) .& (valid_labels .== 1))
    tn = sum((res .== 0) .& (valid_labels .== 0))
    fp = sum((res .== 1) .& (valid_labels .== 0))
    fn = sum((res .== 0) .& (valid_labels .== 1))

	if verbose
       println("--- k-TSP evaluation ---")
       println("  ACC: $(round(acc, digits=4))")
       println("  MCC: $(round(mcc, digits=4))")
       println("  Confusion: [TP=$tp, FP=$fp; FN=$fn, TN=$tn]")
	end

    return (acc=acc, mcc=mcc, preds=res)
end

"""
    evaluate_auctsp(model, valid_data, valid_gene_ids, valid_labels; verbose = false)

Return accuracy, MCC, and prediction result for a trained AUCTSP model
on a validation dataset.
"""
function evaluate_auctsp(
    model::AUCTSPModel,
    valid_data::Matrix{Float64},
    valid_gene_ids::Vector,
    valid_labels::AbstractVector;
	verbose::Bool = false
)
    res = predict_auctsp(model, valid_data, valid_gene_ids)

    acc = mean(res .== valid_labels)
    mcc = _calculate_mcc(res, valid_labels)

    tp = sum((res .== 1) .& (valid_labels .== 1))
    tn = sum((res .== 0) .& (valid_labels .== 0))
    fp = sum((res .== 1) .& (valid_labels .== 0))
    fn = sum((res .== 0) .& (valid_labels .== 1))

	if verbose
       println("--- AUC-TSP evaluation ---")
       println("  ACC: $(round(acc, digits=4))")
       println("  MCC: $(round(mcc, digits=4))")
       println("  Confusion: [TP=$tp, FP=$fp; FN=$fn, TN=$tn]")
	end

    return (acc=acc, mcc=mcc, preds=res)
end
