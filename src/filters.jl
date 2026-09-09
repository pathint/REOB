# src/filters.jl

using Statistics, StatsBase, HypothesisTests, ThreadsX, Combinatorics

"""
    preprocess_filters(data, labels, gene_ids, cfg, confounders=nothing) -> (final_pairs, X)

Run the full pre-filtering pipeline: low-expression removal, differential
ranking, BQC pair filtering, optional confounding-factor audit, hub-gene
pruning, feature-matrix construction, and correlation pruning.
"""
function preprocess_filters(
    data::Matrix{<:Real},
    labels::AbstractVector,
    gene_ids::Vector,
    cfg::REOConfig,
    confounders::Union{Nothing,Vector{<:AbstractVector}}=nothing,
)
    # 1. Filter low-expression genes
    cfg.verbose && println(">>> Filtering low-expression genes...")
    keep_low = filter_low_rank_genes(data, cfg.low_rank_q; verbose=cfg.verbose)

    # 2. Differential rank filter
    cfg.verbose && println(">>> Filtering by differential rank...")
    selected_genes = filter_diff_rank_genes(
        data, labels, keep_low; top_n=cfg.top_diff_n, verbose=cfg.verbose
    )

    # 3. Generate all candidate gene pairs
    all_pairs = collect(combinations(selected_genes, 2))
    cfg.verbose && println(">>> $(length(all_pairs)) candidate pairs generated.")

    # 3.1 BQC stability audit
    pairs_initial = filter_pairs_by_bqc(all_pairs, data, labels, keep_low, cfg)
    cfg.verbose &&
        println(">>> After BQC filtering: $(length(pairs_initial)) pairs remain.")

    isempty(pairs_initial) && error(
        "No gene pairs remain after BQC. Relax bqc_threshold or p0_threshold and retry."
    )

    # 4. Confounding-factor audit
    if !isnothing(confounders) && !isempty(confounders)
        cfg.verbose && println(">>> Auditing against $(length(confounders)) confounders...")

        keep_mask = fill(true, length(pairs_initial))
        p_val_cutoff = cfg.p_val_cutoff

        for (i, pair) in enumerate(pairs_initial)
            g1_idx, g2_idx = pair
            reo_vec = data[g1_idx, :] .> data[g2_idx, :]

            for cf_vec in confounders
                if is_confounded(reo_vec, cf_vec, p_val_cutoff)
                    keep_mask[i] = false
                    break
                end
            end
        end

        n_removed = count(!, keep_mask)
        pairs_initial = pairs_initial[keep_mask]
        cfg.verbose && println("    Removed $n_removed pairs correlated with covariates.")
    end

    # 5. Hub-gene pruning
    pairs_pruned = prune_hub_genes(pairs_initial, cfg.max_occurrence; verbose=cfg.verbose)

    # 6. Build direction-aligned binary feature matrix
    X_initial, pairs_pruned = build_feature_matrix_aligned(data, pairs_pruned, labels)

    # 7. Correlation pruning
    final_pairs, X_final = drop_correlated_features(
        X_initial, pairs_pruned, cfg.cor_threshold
    )

    return final_pairs, X_final
end

"""
    filter_low_rank_genes(data, threshold=0.2; verbose=false) -> Vector{Int}

Remove genes whose median within-sample percentile rank falls below `threshold`.
"""
function filter_low_rank_genes(data::Matrix{Float64}, threshold=0.2; verbose=false)
    n_genes, n_samples = size(data)
    percentile_ranks = Matrix{Float64}(undef, n_genes, n_samples)
    Threads.@threads for j in 1:n_samples
        percentile_ranks[:, j] .= tiedrank(data[:, j]) ./ n_genes
    end

    keep_indices = findall(i -> median(percentile_ranks[i, :]) > threshold, 1:n_genes)
    verbose &&
        println("  Low-expression filter: kept $(length(keep_indices)) / $n_genes genes.")
    return keep_indices
end

"""
    filter_diff_rank_genes(data, labels, gene_indices; top_n=500, verbose=false) -> Vector{Int}

Retain the `top_n` genes with the largest absolute mean percentile-rank
difference between the two classes.
"""
function filter_diff_rank_genes(
    data::Matrix{Float64},
    labels::AbstractVector,
    gene_indices::Vector{Int};
    top_n=500,
    verbose=false,
)
    n_samples = size(data, 2)
    n_genes_subset = length(gene_indices)

    sub_data = data[gene_indices, :]
    percentile_ranks = Matrix{Float64}(undef, n_genes_subset, n_samples)
    for j in 1:n_samples
        percentile_ranks[:, j] .= tiedrank(sub_data[:, j]) ./ n_genes_subset
    end

    idx1 = findall(==(1), labels)
    idx0 = findall(==(0), labels)

    diffs = [
        abs(mean(percentile_ranks[i, idx1]) - mean(percentile_ranks[i, idx0])) for
        i in 1:n_genes_subset
    ]

    p = sortperm(diffs; rev=true)
    selected_internal_indices = p[1:min(top_n, length(p))]

    final_indices = gene_indices[selected_internal_indices]
    verbose && println("  Differential filter: kept $(length(final_indices)) genes.")
    return final_indices
end

"""
    get_top_pairs_parallel_fisher(data, labels, gene_indices; n_top=5000, verbose=false)

Parallel Fisher exact test to rank gene pairs by discriminative power.
"""
function get_top_pairs_parallel_fisher(
    data::Matrix{Float64},
    labels::AbstractVector,
    gene_indices::Vector{Int};
    n_top=5000,
    verbose=false,
)
    idx1 = findall(==(1), labels)
    idx0 = findall(==(0), labels)
    n1, n0 = length(idx1), length(idx0)

    all_pairs = collect(combinations(gene_indices, 2))
    n_pairs = length(all_pairs)
    p_values = Vector{Float64}(undef, n_pairs)

    Threads.@threads for i in 1:n_pairs
        g1, g2 = all_pairs[i]
        c1 = sum(data[g1, idx1] .> data[g2, idx1])
        c0 = sum(data[g1, idx0] .> data[g2, idx0])

        # 2x2 contingency table:
        #          g1>g2  g1<=g2
        # Label 1:  c1    n1-c1
        # Label 0:  c0    n0-c0
        ft = FisherExactTest(c1, n1-c1, c0, n0-c0)
        p_values[i] = pvalue(ft)
    end

    sp = sortperm(p_values)
    top_indices = sp[1:min(n_top, n_pairs)]

    verbose && println("  Fisher exact test: retained $(min(n_top, n_pairs)) pairs.")
    return all_pairs[top_indices], p_values[top_indices]
end

"""
    filter_pairs_by_bqc(pairs, data, labels, keep_low, cfg) -> Vector

Rank gene pairs by the enhanced Bayesian Quality Control score.  Pairs must
pass both the BQC threshold and the p0 stability threshold.
"""
function filter_pairs_by_bqc(pairs, data, labels, keep_low, cfg::REOConfig)
    idx0 = findall(==(0), labels)
    idx1 = findall(==(1), labels)
    n0 = length(idx0)
    n1 = length(idx1)

    # 1. Estimate global tau
    cfg.verbose && println(">>> Estimating global tau...")
    tau_res = estimate_global_tau_parallel(data[keep_low, :])
    tau = tau_res.mean

    bqc_threshold = cfg.bqc_threshold
    p0_threshold = cfg.p0_threshold
    threshold_dict = generate_bqc_threshold_dict(n0, n1, tau, bqc_threshold, p0_threshold)
    cfg.verbose &&
        println(">>> BQC threshold dictionary: $(length(threshold_dict)) entries.")
    pairs_initial = filter_pairs_with_dict(
        pairs, data, labels, threshold_dict; verbose=cfg.verbose
    )

    n_pairs = length(pairs_initial)
    results = Vector{
        NamedTuple{(:pair, :score, :p0_diff),Tuple{Tuple{Int,Int},Float64,Float64}}
    }(
        undef, n_pairs
    )

    cfg.verbose && println(">>> Computing BQC scores for $n_pairs pairs...")

    # 2. Parallel BQC scoring
    Threads.@threads for i in 1:n_pairs
        g1, g2 = pairs_initial[i]

        @views k0 = sum(data[g1, idx0] .> data[g2, idx0])
        p0 = k0 / n0
        p0_diff = abs(p0 - 0.5)

        @views k1 = sum(data[g1, idx1] .> data[g2, idx1])

        score = calculate_enhanced_bqc(k1, n1, p0, tau)

        results[i] = (pair=(g1, g2), score=score, p0_diff=p0_diff)
    end

    # 3. Two-level sort: primary by score (desc), secondary by p0_diff (desc)
    sort!(results; by=x -> (x.score, x.p0_diff), rev=true)

    sorted_pairs = [x.pair for x in results]

    return sorted_pairs
end

"""
    filter_pairs_with_dict(pairs, data, labels, threshold_dict; verbose = false)

Peform a fast gene pairs filtering step with a pre-calculated dictionary of thresholds.
"""
function filter_pairs_with_dict(pairs, data, labels, threshold_dict::Dict; verbose=false)
    idx0 = findall(==(0), labels)
    idx1 = findall(==(1), labels)
    n0 = length(idx0) / 2

    keep_mask = fill(false, length(pairs))

    Threads.@threads for i in 1:length(pairs)
        g1, g2 = pairs[i]
        k0 = sum(data[g1, idx0] .> data[g2, idx0])

        limit = get(threshold_dict, k0, nothing)

        if !isnothing(limit)
            k1 = sum(data[g1, idx1] .> data[g2, idx1])
            if (k0 < n0 && k1 >= limit) || (k0 > n0 && k1 <= limit)
                keep_mask[i] = true
            end
        end
    end

    verbose && println("    BQC dictionary filter: retained $(sum(keep_mask)) pairs.")
    return pairs[keep_mask]
end

"""
    prune_hub_genes(pairs, max_occurrence=2; verbose=false)

Prevent any single gene from appearing in more than `max_occurrence` pairs.
"""
function prune_hub_genes(pairs, max_occurrence=2; verbose=false)
    gene_counts = Dict{Int,Int}()
    final_pairs = []

    for (g1, g2) in pairs
        c1 = get(gene_counts, g1, 0)
        c2 = get(gene_counts, g2, 0)

        if c1 < max_occurrence && c2 < max_occurrence
            push!(final_pairs, (g1, g2))
            gene_counts[g1] = c1 + 1
            gene_counts[g2] = c2 + 1
        end
    end

    verbose && println("  After hub pruning: $(length(final_pairs)) pairs remain.")
    return final_pairs
end

"""
    build_feature_matrix(data, pairs) -> BitMatrix

Convert gene-pair orderings (A > B) into a binary feature matrix.
"""
function build_feature_matrix(data::Matrix{Float64}, pairs)
    n_samples = size(data, 2)
    n_pairs = length(pairs)
    X = BitArray(undef, (n_samples, n_pairs))

    Threads.@threads for j in 1:n_pairs
        g1, g2 = pairs[j]
        @views X[:, j] .= data[g1, :] .> data[g2, :]
    end
    return X
end

"""
    build_feature_matrix_aligned(data, pairs, labels) -> (X, new_pairs)

Build a binary feature matrix with direction aligned so that `g1 > g2`
correlates with the positive class.  Returns the matrix and the (possibly
flipped) pair indices.
"""
function build_feature_matrix_aligned(data::Matrix{Float64}, pairs, labels::AbstractVector)
    n_samples = size(data, 2)
    n_pairs = length(pairs)

    X = BitArray(undef, (n_samples, n_pairs))
    new_pairs = Vector{Tuple{Int,Int}}(undef, n_pairs)

    pos_idx = findall(==(1), labels)
    neg_idx = findall(==(0), labels)
    n_pos = length(pos_idx)
    n_neg = length(neg_idx)

    Threads.@threads for j in 1:n_pairs
        g1, g2 = pairs[j]

        @views p_pos = sum(data[g1, pos_idx] .> data[g2, pos_idx]) / n_pos
        @views p_neg = sum(data[g1, neg_idx] .> data[g2, neg_idx]) / n_neg

        if p_pos < p_neg
            new_pairs[j] = (g2, g1)
        else
            new_pairs[j] = (g1, g2)
        end

        @views X[:, j] .= data[new_pairs[j][1], :] .> data[new_pairs[j][2], :]
    end

    return X, new_pairs
end

"""
    drop_correlated_features(X, pairs, threshold=0.95) -> (pairs, X)

Remove features whose pairwise Pearson correlation exceeds `threshold`.
"""
function drop_correlated_features(X::AbstractMatrix, pairs, threshold=0.95)
    n_features = size(X, 2)
    keep = trues(n_features)
    cor_mat = cor(X)

    for i in 1:n_features
        !keep[i] && continue
        for j in (i + 1):n_features
            if keep[j] && abs(cor_mat[i, j]) > threshold
                keep[j] = false
            end
        end
    end

    return pairs[keep], X[:, keep]
end

"""
    is_confounded(reo_vec, cf_vec, p_threshold) -> Bool

Test whether a gene-pair ordering is significantly associated with a
confounding variable (continuous → Welch t-test, categorical → chi-squared).
"""
function is_confounded(reo_vec::BitVector, cf_vec::AbstractVector, p_threshold::Float64)
    if eltype(cf_vec) <: AbstractFloat
        group0 = cf_vec[reo_vec .== 0]
        group1 = cf_vec[reo_vec .== 1]

        length(group0) < 5 || length(group1) < 5 && return false
        std(group0) ≈ 0 && std(group1) ≈ 0 && return false

        return pvalue(UnequalVarianceTTest(group0, group1)) < p_threshold
    else
        tbl = counts(reo_vec, cf_vec)
        try
            return pvalue(ChisqTest(tbl)) < p_threshold
        catch
            return false
        end
    end
end

"""
    filter_genes(data, labels, gene_ids, cfg) -> Vector{Int}

Gene-level pre-filtering for TSP-family methods (low-expression + differential
rank filtering only, no BQC).
"""
function filter_genes(
    data::Matrix{<:Real}, labels::AbstractVector, gene_ids::Vector, cfg::REOConfig
)
    keep_low = filter_low_rank_genes(data, cfg.low_rank_q; verbose=cfg.verbose)
    selected_genes = filter_diff_rank_genes(
        data, labels, keep_low; top_n=cfg.top_diff_n, verbose=cfg.verbose
    )
    return selected_genes
end
