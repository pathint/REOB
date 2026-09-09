using Distributions, QuadGK, Statistics
using Base.Threads
using ThreadsX

"""
    estimate_global_tau(data) -> Float64

Estimate the global signal-to-noise ratio tau from the expression matrix.
"""
function estimate_global_tau(data::Matrix{Float64})
    gene_means = mean(data; dims=2)
    S = std(gene_means)

    n_genes = size(data, 1)
    sample_size = min(1000, n_genes)
    indices = rand(1:n_genes, sample_size)

    pair_stds = [
        std(data[indices[i], :] .- data[indices[j], :]) for i in 1:sample_size for
        j in (i + 1):sample_size
    ]
    sigma_D = median(pair_stds)

    return max((sqrt(2) * S) / sigma_D, 2.0)
end

"""
    estimate_global_tau_parallel(data; n_iters=20, sample_size=1000) -> (mean, std, all_values)

Parallel estimation of tau via repeated random sub-sampling.
"""
function estimate_global_tau_parallel(
    data::Matrix{Float64}, n_iters::Int=20; sample_size::Int=1000
)
    n_genes = size(data, 1)

    gene_means = mean(data; dims=2)
    S = std(gene_means)

    tau_results = zeros(Float64, n_iters)
    actual_sample_size = min(sample_size, n_genes)

    Threads.@threads for k in 1:n_iters
        indices = rand(1:n_genes, actual_sample_size)
        n_pairs = Int(actual_sample_size * (actual_sample_size - 1) / 2)
        pair_stds = Vector{Float64}(undef, n_pairs)

        idx = 1
        for i in 1:actual_sample_size
            for j in (i + 1):actual_sample_size
                pair_stds[idx] = std(data[indices[i], :] .- data[indices[j], :])
                idx += 1
            end
        end

        sigma_D = median(pair_stds)
        tau_results[k] = max((sqrt(2) * S) / sigma_D, 2.0)
    end

    mean_tau = mean(tau_results)
    std_tau = std(tau_results)

    return (mean=mean_tau, std=std_tau, all_values=tau_results)
end

"""
    calculate_bayesian_shift_score(k, n, p0, tau) -> Float64

Compute the posterior probability that the ordering has flipped relative to
the control-group baseline p0, given a U-shaped Beta-like prior scaled by tau.
"""
function calculate_bayesian_shift_score(k, n, p0, tau)
    prior(p) = begin
        if p <= 1e-6 || p >= 1-1e-6
            return 0.0
        end
        z = quantile(Normal(), p)
        return (1/tau) * exp(z^2 * 0.5 * (1 - 1/tau^2))
    end

    likelihood(p) = pdf(Binomial(n, p), k)

    numerator, _ = quadgk(p -> likelihood(p) * prior(p), 0.5, 1.0)
    denominator, _ = quadgk(p -> likelihood(p) * prior(p), 0.0, 1.0)

    post_prob = numerator / (denominator + 1e-12)
    return p0 < 0.5 ? post_prob : (1.0 - post_prob)
end

"""
    calculate_enhanced_bqc(k1, n1, p0, tau) -> Float64

Enhanced Bayesian Quality Control score.  Combines the posterior shift
probability with an anchor weight that penalises pairs whose control-group
ordering is close to 0.5 (unstable).
"""
function calculate_enhanced_bqc(k1, n1, p0, tau)
    conf = calculate_bayesian_shift_score(k1, n1, p0, tau)

    anchor_weight = abs(p0 - 0.5) * 2.0
    eps = 1e-15
    nlp_score = -log10(1.0 - conf + eps)

    return nlp_score * anchor_weight
end

"""
    generate_bqc_lookup_table(n0, n1, tau) -> Matrix{Float64}

Pre-compute the enhanced-BQC score for every (k0, k1) combination.
"""
function generate_bqc_lookup_table(n0::Int, n1::Int, tau::Real)
    table = zeros(Float64, n0 + 1, n1 + 1)

    Threads.@threads for k0 in 0:n0
        p0 = k0 / n0
        for k1 in 0:n1
            table[k0 + 1, k1 + 1] = calculate_enhanced_bqc(k1, n1, p0, tau)
        end
    end

    return table
end

"""
    lookup_bqc(table, k0, k1) -> Float64

Fast O(1) lookup into a pre-computed BQC table.
"""
@inline function lookup_bqc(table::Matrix{Float64}, k0::Int, k1::Int)
    return table[k0 + 1, k1 + 1]
end

"""
    generate_bqc_threshold_dict(n0, n1, tau, bqc_limit, p0_threshold) -> Dict{Int,Int}

Build a lookup dictionary mapping each valid k0 to the critical k1 that
achieves the BQC score threshold.  Only k0 values satisfying the p0 stability
criterion are included.
"""
function generate_bqc_threshold_dict(
    n0::Int, n1::Int, tau::Real, bqc_limit::Float64, p0_threshold::Float64
)
    threshold_dict = Dict{Int,Int}()

    k0_high_min = ceil(Int, (0.5 + p0_threshold) * n0)
    k0_low_max = floor(Int, (0.5 - p0_threshold) * n0)

    for k0 in 0:k0_low_max
        p0 = k0 / n0
        pre_k1 = get(threshold_dict, k0 - 1, nothing)
        fro_k1 = isnothing(pre_k1) ? ceil(Int, n1/2) : pre_k1
        for k1 in fro_k1:n1
            if calculate_enhanced_bqc(k1, n1, p0, tau) >= bqc_limit
                threshold_dict[k0] = k1
                threshold_dict[n0 - k0] = n1 - k1  # symmetry
                break
            end
        end
    end

    return threshold_dict
end

"""
    calibrate_threshold(scores, labels) -> Float64

Find the optimal classification threshold using Youden's Index.
"""
function calibrate_threshold(scores, labels)
    thresholds = sort(unique(scores))
    length(thresholds) < 2 && return 0.5

    best_j = -1.0
    best_t = 0.5

    for t in thresholds
        preds = scores .>= t
        tp = sum((preds .== 1) .& (labels .== 1))
        tn = sum((preds .== 0) .& (labels .== 0))
        fp = sum((preds .== 1) .& (labels .== 0))
        fn = sum((preds .== 0) .& (labels .== 1))

        sens = tp / (tp + fn + 1e-9)
        spec = tn / (tn + fp + 1e-9)
        j = sens + spec - 1

        if j > best_j
            best_j = j
            best_t = t
        end
    end
    return best_t
end
