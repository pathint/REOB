# src/training.jl

using Lasso, DecisionTree, Random, Statistics
using IterTools, Statistics, DataFrames

"""
    fit_reo(data, labels, gene_ids, cfg; confounders=nothing) -> REOModel

Main entry point for training an REOB model.  Runs the shared pre-filtering
pipeline then dispatches to the chosen strategy (Voting / RF / Lasso).
"""
function fit_reo(
    data::Matrix{Float64},
    labels::AbstractVector,
    gene_ids::Vector,
    cfg::REOConfig;
    confounders::Union{Nothing,Vector{<:AbstractVector}}=nothing,
)

    # Phase 1: shared pre-filtering
    final_pairs_idx, X = preprocess_filters(data, labels, gene_ids, cfg, confounders)

    isempty(final_pairs_idx) &&
        error("No valid gene pairs after pre-filtering. Relax filtering parameters.")

    # Phase 2: method-specific training
    initial_model = if cfg.method == VotingMethod
        if length(final_pairs_idx) > 128
            final_pairs_idx = final_pairs_idx[1:128]
            X = X[:, 1:128]
        end
        _fit_voting_strategy(X, labels, gene_ids, final_pairs_idx, cfg)
    elseif cfg.method == RFMethod
        _fit_rf_strategy(X, labels, gene_ids, final_pairs_idx, cfg)
    elseif cfg.method == LassoMethod
        _fit_lasso_strategy(X, labels, gene_ids, final_pairs_idx, cfg)
    else
        error("Unknown method: $(cfg.method)")
    end

    return initial_model
end

# =============================================================================
# Random Forest strategy: stability selection + orientation alignment
# =============================================================================
"Peform random forest model train pipeline."
function _fit_rf_strategy(X, labels, gene_ids, pairs_idx, cfg)
    cfg.verbose && println(">>> Running RF (stumps) stability selection...")

    scores, top_10 = select_top_10_models(
        X, labels, cfg.ss_iterations, cfg.ss_ratio, cfg.target_n; verbose=cfg.verbose
    )

    final_model = top_10[1].model
    raw_weights = impurity_importance(final_model)

    # Orientation alignment: ensure g1 > g2 always means positive class
    final_named_pairs = Vector{Tuple{String,String}}()
    aligned_weights = Float64[]

    top_indices = unique([tree.featid for tree in final_model.trees if isa(tree, Node)])
    raw_weights = raw_weights[top_indices]
    for (i, idx) in enumerate(top_indices)
        g1_idx, g2_idx = pairs_idx[idx]
        p_rate = mean(X[labels .== 1, idx])
        n_rate = mean(X[labels .== 0, idx])

        if p_rate >= n_rate
            push!(final_named_pairs, (gene_ids[g1_idx], gene_ids[g2_idx]))
        else
            push!(final_named_pairs, (gene_ids[g2_idx], gene_ids[g1_idx]))
        end
        push!(aligned_weights, raw_weights[i])
    end

    norm_weights = aligned_weights ./ sum(aligned_weights)
    intercept = 0

    return REOModel(cfg, final_named_pairs, norm_weights, intercept)
end

"""
    stability_selection_rf_stumps(X, y, n_iterations, sub_sample_ratio; ...) -> scores

Stability selection using decision stumps with internal permutation testing
and OOB performance weighting.
"""
function stability_selection_rf_stumps(
    X,
    y,
    n_iterations=500,
    sub_sample_ratio=0.8;
    n_perms=100,
    p_threshold=0.05,
    verbose=false,
)
    n_samples, n_features = size(X)
    all_indices = 1:n_samples
    idx1 = findall(==(1), y)
    idx0 = findall(==(0), y)

    selection_scores = zeros(Float64, n_features)
    lk = ReentrantLock()

    verbose && println("......Starting RF stump stability selection...")

    Threads.@threads for i in 1:n_iterations
        # Stratified sub-sampling
        sub_idx = vcat(
            sample(idx1, Int(floor(length(idx1)*sub_sample_ratio)), replace=true),
            sample(idx0, Int(floor(length(idx0)*sub_sample_ratio)), replace=true),
        )
        oob_idx = setdiff(all_indices, sub_idx)

        X_sub, y_sub = X[sub_idx, :], y[sub_idx]
        X_oob, y_oob = X[oob_idx, :]

        model = build_forest(y_sub, X_sub, floor(Int, sqrt(n_features)), 50, 0.7, 1)

        selected_feats = unique([tree.featid for tree in model.trees if isa(tree, Node)])
        isempty(selected_feats) && continue

        # Internal permutation test
        sub_preds = apply_forest(model, X_sub)
        obs_sub_mcc = _calculate_mcc(sub_preds .> 0.5, y_sub)

        hit_count = 0
        y_sub_perm = copy(y_sub)
        X_view = X_sub[:, selected_feats]
        for _ in 1:n_perms
            shuffle!(y_sub_perm)
            perm_mcc = _quick_refit_mcc(X_view, y_sub_perm)
            if perm_mcc >= obs_sub_mcc
                hit_count += 1
            end
        end
        p_val = (hit_count + 1) / (n_perms + 1)

        if p_val <= p_threshold
            oob_preds = apply_forest(model, X_oob)
            oob_mcc = _calculate_mcc(oob_preds .> 0.5, y_oob)
            if oob_mcc > 0
                weight = 1 / (1 + exp(-oob_mcc))

                lock(lk) do
                    for tree in model.trees
                        if isa(tree, Node)
                            feat_idx = tree.featid
                            selection_scores[feat_idx] += weight
                        end
                    end
                end
            end
        end
    end

    max_s = maximum(selection_scores)
    return max_s > 0 ? selection_scores ./ max_s : selection_scores
end

"""
Quick refit: simulate optimal voting rule under permuted labels.
"""
function _quick_refit_mcc(X_view, y_perm)
    n_samples, n_f = size(X_view)
    scores = zeros(Float64, n_samples)
    for j in 1:n_f
        acc_pos = mean(X_view[:, j] .== y_perm)
        direction = acc_pos >= 0.5 ? 1.0 : -1.0

        if direction > 0
            scores .+= X_view[:, j]
        else
            scores .+= (1 .- X_view[:, j])
        end
    end
    return _calculate_mcc((scores ./ n_f) .> 0.5, y_perm)
end

"""
Performance-filtered stability selection: retains only elite models whose
OOB performance exceeds the 75th percentile.
"""
function stability_selection_performance_filtered(
    X, y, n_iterations=500, sub_sample_ratio=0.8; verbose=false
)
    n_samples, n_features = size(X)
    idx1 = findall(==(1), y);
    idx0 = findall(==(0), y)
    all_indices = 1:n_samples

    iter_results = Vector{Any}(undef, n_iterations)

    verbose && println(">>> Starting performance-audited RF-stumps stability selection...")

    Threads.@threads for i in 1:n_iterations
        sub_idx = vcat(
            sample(idx1, Int(floor(length(idx1)*sub_sample_ratio)), replace=false),
            sample(idx0, Int(floor(length(idx0)*sub_sample_ratio)), replace=false),
        )
        oob_idx = setdiff(all_indices, sub_idx)

        X_sub, y_sub = X[sub_idx, :], y[sub_idx]
        X_oob, y_oob = X[oob_idx, :], y[oob_idx]

        model = build_forest(y_sub, X_sub, floor(Int, sqrt(n_features)), 50, 0.7, 1)

        train_preds = apply_forest(model, X_sub)
        oob_preds = apply_forest(model, X_oob)

        train_mcc = _calculate_mcc(train_preds .> 0.5, y_sub)
        oob_mcc = _calculate_mcc(oob_preds .> 0.5, y_oob)

        feat_hits = Dict{Int,Float64}()
        for tree in model.trees
            if isa(tree, Node)
                feat_hits[tree.featid] = get(feat_hits, tree.featid, 0.0) + 1.0
            end
        end

        iter_results[i] = (train_mcc=train_mcc, oob_mcc=oob_mcc, feat_hits=feat_hits)
    end

    train_mccs = [r.train_mcc for r in iter_results]
    oob_mccs = [r.oob_mcc for r in iter_results]

    oob_threshold = quantile(oob_mccs, 0.75)

    selection_scores = zeros(Float64, n_features)
    valid_model_count = 0

    for r in iter_results
        if r.oob_mcc > max(0.1, oob_threshold)
            valid_model_count += 1
            w = r.oob_mcc^2
            for (f_idx, hit) in r.feat_hits
                selection_scores[f_idx] += hit * w
            end
        end
    end

    verbose &&
        println(">>> Audit complete: $valid_model_count / $n_iterations models passed.")

    max_s = maximum(selection_scores)
    return max_s > 0 ? selection_scores ./ max_s : selection_scores
end

"""
Generalization-focused stability selection: selects models with the smallest
train–OOB performance gap.
"""
function stability_selection_generalization(
    X, y, n_iterations=500, sub_sample_ratio=0.7; verbose=true
)
    n_samples, n_features = size(X)
    idx1 = findall(==(1), y);
    idx0 = findall(==(0), y)

    performance_log = DataFrame(;
        iter=Int[], train_mcc=Float64[], oob_mcc=Float64[], gap=Float64[], feat_ids=Any[]
    )
    selection_scores = zeros(Float64, n_features)
    lk = ReentrantLock()

    verbose && println(">>> Starting generalization audit (gap minimization)...")

    Threads.@threads for i in 1:n_iterations
        sub_idx = vcat(
            sample(idx1, Int(floor(length(idx1)*sub_sample_ratio)), replace=false),
            sample(idx0, Int(floor(length(idx0)*sub_sample_ratio)), replace=false),
        )
        oob_idx = setdiff(1:n_samples, sub_idx)

        model = build_forest(
            y[sub_idx], X[sub_idx, :], floor(Int, sqrt(n_features)), 50, 0.7, 1
        )

        t_preds = apply_forest(model, X[sub_idx, :])
        o_preds = apply_forest(model, X[oob_idx, :])

        tmcc = _calculate_mcc(t_preds .> 0.5, y[sub_idx])
        omcc = _calculate_mcc(o_preds .> 0.5, y[oob_idx])

        gap = max(0.0, tmcc - omcc)
        f_ids = [tree.featid for tree in model.trees if isa(tree, Node)]

        lock(lk) do
            push!(performance_log, (i, tmcc, omcc, gap, f_ids))
        end
    end

    base_line = quantile(performance_log.train_mcc, 0.5)
    gap_threshold = quantile(performance_log.gap, 0.25)

    elite_models = filter(
        r -> r.train_mcc >= base_line && r.gap <= gap_threshold, performance_log
    )

    verbose && println(
        ">>> Audit complete: $(nrow(elite_models)) / $(nrow(performance_log)) low-gap models selected.",
    )

    for row in eachrow(elite_models)
        weight = row.oob_mcc * (1.0 / (1.0 + row.gap))
        for fid in row.feat_ids
            selection_scores[fid] += max(0.0, weight)
        end
    end

    max_s = maximum(selection_scores)
    final_scores = max_s > 0 ? selection_scores ./ max_s : selection_scores
    return final_scores
end

"""
    select_top_10_models(X, y, n_iterations, sub_sample_ratio, target_n) -> (scores, top_10)

Train `n_iterations` stump forests and return stability scores together with
the 10 highest-OOB-MCC models.
"""
function select_top_10_models(
    X, y, n_iterations=500, sub_sample_ratio=0.7, target_n=50; verbose=true
)
    n_samples, n_features = size(X)
    idx1 = findall(==(1), y);
    idx0 = findall(==(0), y)

    all_results = Vector{Any}(undef, n_iterations)
    lk = ReentrantLock()

    verbose && println(">>> Training $n_iterations stump forests to select top 10...")

    Threads.@threads for i in 1:n_iterations
        sub_idx = vcat(
            sample(idx1, Int(floor(length(idx1)*sub_sample_ratio)), replace=false),
            sample(idx0, Int(floor(length(idx0)*sub_sample_ratio)), replace=false),
        )
        oob_idx = setdiff(1:n_samples, sub_idx)

        model = build_forest(y[sub_idx], X[sub_idx, :], -1, target_n, 0.7, 1)

        t_preds = apply_forest(model, X[sub_idx, :])
        o_preds = apply_forest(model, X[oob_idx, :])

        tmcc = _calculate_mcc(t_preds .> 0.5, y[sub_idx])
        omcc = _calculate_mcc(o_preds .> 0.5, y[oob_idx])

        all_results[i] = (model=model, train_mcc=tmcc, oob_mcc=omcc)
    end

    valid_results = filter(x -> !isnothing(x) && !isnan(x.oob_mcc), all_results)
    sort!(valid_results; by=x -> x.oob_mcc, rev=true)
    top_10 = valid_results[1:min(10, length(valid_results))]

    selection_scores = zeros(Float64, n_features)
    for res in top_10
        w = res.oob_mcc
        for tree in res.model.trees
            if isa(tree, Node)
                selection_scores[tree.featid] += w
            end
        end
    end

    max_s = maximum(selection_scores)
    final_scores = max_s > 0 ? selection_scores ./ max_s : selection_scores

    return final_scores, top_10
end

# =============================================================================
# Lasso strategy: Elastic Net stability selection
# =============================================================================

"Peform Lasso model train pipeline."
function _fit_lasso_strategy(X, labels, gene_ids, pairs_idx, cfg)
    cfg.verbose &&
        println(">>> Running Lasso path with OOB-weighted stability selection...")

    sel_scores = stability_selection_lasso(X, labels, cfg)

    stable_idx = findall(p -> p >= cfg.ss_threshold, sel_scores)
    if isempty(stable_idx)
        cfg.verbose &&
            println("......No features above threshold; taking top $(cfg.target_n*2)...")
        stable_idx = sortperm(sel_scores; rev=true)[1:min(
            cfg.target_n * 2, length(sel_scores)
        )]
    end

    X_stable = X[:, stable_idx]
    pairs_stable = pairs_idx[stable_idx]

    n_samples = length(labels)
    n_pos = sum(labels .== 1);
    n_neg = sum(labels .== 0)
    final_wts = [l == 1 ? (n_samples/(2*n_pos)) : (n_samples/(2*n_neg)) for l in labels]

    final_path = fit(
        LassoPath,
        X_stable,
        Float64.(labels),
        Binomial(),
        LogitLink();
        wts=final_wts,
        standardize=true,
        intercept=true,
        α=0.9,
        irls_maxiter=200,
        irls_tol=1e-4,
        λminratio=0.1,
    )

    raw_weights, raw_intercept, active_lasso_idx = extract_model_params(
        final_path, cfg.target_n
    )

    # Orientation alignment
    final_named_pairs = Vector{Tuple{String,String}}()
    positive_weights = Float64[]
    adjusted_intercept = raw_intercept

    for i in 1:length(active_lasso_idx)
        idx_in_stable = active_lasso_idx[i]
        w = raw_weights[i]
        p = pairs_stable[idx_in_stable]
        g1_name, g2_name = gene_ids[p[1]], gene_ids[p[2]]

        if w > 0
            push!(final_named_pairs, (g1_name, g2_name))
            push!(positive_weights, w)
        else
            push!(final_named_pairs, (g2_name, g1_name))
            push!(positive_weights, abs(w))
            adjusted_intercept += w
        end
    end

    total_w = sum(positive_weights)
    final_weights = positive_weights ./ total_w
    final_intercept = adjusted_intercept / total_w

    return REOModel(cfg, final_named_pairs, final_weights, final_intercept)
end

"""
Extract coefficients at the lambda position where `target_n` features are active.
"""
function extract_model_params(model_path, target_n)
    coef_matrix = model_path.coefs
    n_nonzero = [count(!iszero, coef_matrix[:, i]) for i in 1:size(coef_matrix, 2)]

    best_idx = findfirst(x -> x >= target_n, n_nonzero)
    isnothing(best_idx) && (best_idx = size(coef_matrix, 2))

    full_weights = coef_matrix[:, best_idx]
    active_idx = findall(!iszero, full_weights)

    return Vector(full_weights[active_idx]), model_path.b0[best_idx], active_idx
end

"""
OOB-weighted Lasso stability selection.
"""
function stability_selection_lasso(X, y, cfg::REOConfig)
    n_samples, n_features = size(X)
    all_idx = 1:n_samples

    idx1 = findall(x -> x == 1, y)
    idx0 = findall(x -> x == 0, y)

    n1_sub = max(Int(floor(length(idx1) * cfg.ss_ratio)), 3)
    n0_sub = max(Int(floor(length(idx0) * cfg.ss_ratio)), 3)

    selection_scores = zeros(Float64, n_features)
    lk = ReentrantLock()

    cfg.verbose && println(
        "Starting OOB-weighted stability selection ($(cfg.ss_iterations) iterations)..."
    )

    Threads.@threads for i in 1:cfg.ss_iterations
        sub_idx1 = sample(idx1, n1_sub, replace=false)
        sub_idx0 = sample(idx0, n0_sub, replace=false)
        sub_idx = vcat(sub_idx1, sub_idx0)
        oob_idx = setdiff(all_idx, sub_idx)

        X_sub, y_sub = X[sub_idx, :], y[sub_idx]
        X_oob, y_oob = X[oob_idx, :], y[oob_idx]

        col_vars = var(X_sub, dims=1)
        active_cols = [c[2] for c in findall(v -> v > 1e-10, col_vars)]
        length(active_cols) < 2 && continue
        X_filtered = X_sub[:, active_cols]

        n_s = length(y_sub)
        n_s1 = sum(y_sub .== 1);
        n_s0 = sum(y_sub .== 0)
        sub_wts = [l == 1 ? (n_s/(2*n_s1)) : (n_s/(2*n_s0)) for l in y_sub]

        try
            path = fit(
                LassoPath,
                X_filtered,
                Float64.(y_sub),
                Binomial(),
                LogitLink();
                wts=sub_wts,
                standardize=true,
                intercept=true,
                α=0.9,
                irls_maxiter=200,
                irls_tol=1e-4,
                λminratio=0.05,
            )

            mid_idx = size(path.coefs, 2) ÷ 2
            coefs = path.coefs[:, mid_idx]
            intercept = path.b0[mid_idx]

            selected_local = findall(!iszero, coefs)
            isempty(selected_local) && continue
            selected_global = active_cols[selected_local]

            w_active = coefs[selected_local]
            X_oob_active = X_oob[:, selected_global]
            z = (X_oob_active * w_active) .+ intercept
            oob_preds = (1.0 ./ (1.0 .+ exp.(-z))) .> 0.5

            oob_mcc = _calculate_mcc(oob_preds, y_oob)

            if oob_mcc > 0
                lock(lk) do
                    selection_scores[selected_global] .+= oob_mcc
                end
            end
        catch e
            continue
        end
    end

    max_score = maximum(selection_scores)
    return max_score > 0 ? selection_scores ./ max_score : selection_scores
end

# =============================================================================
# Voting strategy: Gray code (≤20 features) or SFFS (≤128 features)
# =============================================================================
"Peform majority voting model train pipeline."
function _fit_voting_strategy(X, labels, gene_ids, pairs_idx, cfg)
    cfg.verbose && println(">>> Running majority-voting feature subset selection...")

    top_indices, err, p_val, tau = select_feature_subset(UInt8.(X), UInt8.(labels))
    cfg.verbose && println(">>> Best threshold for majority vote: $tau")

    n_top = length(top_indices)
    cfg.verbose && println(">>> $n_top gene pairs were selected.")

    final_named_pairs = Vector{Tuple{String,String}}()
    for idx in top_indices
        g1_idx, g2_idx = pairs_idx[idx]
        push!(final_named_pairs, (gene_ids[g1_idx], gene_ids[g2_idx]))
    end

    voting_weights = fill(1.0 / n_top, n_top)
    voting_bias = (cld(n_top, 2) - tau) / n_top

    cfg.verbose && println(final_named_pairs)

    return REOModel(cfg, final_named_pairs, voting_weights, voting_bias)
end
