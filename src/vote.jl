using Random
using Statistics

# =========================================================
# Gray code + exact counting majority vote feature subset
# Correct + fast + brute-force consistent
# =========================================================

# -------------------------
# Feature representation
# -------------------------

"""
    BitFeature

Sparse representation of a single binary feature column.

# Fields
- `idx::Vector{Int}`: Row indices where the feature value equals 1.
"""
struct BitFeature
    idx::Vector{Int}
end

# -------------------------
# Build features (dense → sparse index form)
# -------------------------

"""
    build_features(X::Matrix{UInt8}) -> Vector{BitFeature}

Convert a dense binary matrix `X` (samples × features) into a vector of
`BitFeature` objects, one per column. Each `BitFeature` stores only the row
indices where the column value is 1.
"""
function build_features(X::Matrix{UInt8})
    n, m = size(X)
    feats = Vector{BitFeature}(undef, m)

    for j in 1:m
        idx = Int[]
        for i in 1:n
            if X[i, j] == 1
                push!(idx, i)
            end
        end
        feats[j] = BitFeature(idx)
    end

    return feats
end

# -------------------------
# Feature sorting (important for pruning)
# -------------------------

"""
    sort_features!(feats::Vector{BitFeature}, y::Vector{UInt8}) -> Vector{BitFeature}

Sort features in descending order by a signed score computed as the count of
positive-class rows minus the count of negative-class rows for each feature.
Placing high-scoring features first improves pruning during Gray code search.

Returns the permuted (sorted) feature vector.
"""
function sort_features!(feats, y)
    m = length(feats)
    score = zeros(Float64, m)

    @inbounds for j in 1:m
        s = 0
        for i in feats[j].idx
            s += (y[i] == 1 ? 1 : -1)
        end
        score[j] = s
    end

    perm = sortperm(score; rev=true)
    return feats[perm]
end

# -------------------------
# Add / remove feature (exact update)
# -------------------------

"""
    add_feature!(counts::Vector{Int16}, feat::BitFeature)

Increment the vote count of every sample indicated by `feat.idx` by 1.
Used to maintain running vote tallies when a feature is included in the
current subset during Gray code enumeration.
"""
function add_feature!(counts::Vector{Int16}, feat::BitFeature)
    @inbounds for i in feat.idx
        counts[i] += 1
    end
end

"""
    remove_feature!(counts::Vector{Int16}, feat::BitFeature)

Decrement the vote count of every sample indicated by `feat.idx` by 1.
Used to maintain running vote tallies when a feature is removed from the
current subset during Gray code enumeration.
"""
function remove_feature!(counts::Vector{Int16}, feat::BitFeature)
    @inbounds for i in feat.idx
        counts[i] -= 1
    end
end


"""
    mask_to_indices(mask::Union{UInt64,UInt128}, m::Int) -> Vector{Int}

Convert a bit mask representing a feature subset into a vector of 1-based
feature indices. Bit position `j-1` corresponds to feature index `j`.
"""
function mask_to_indices(mask::Union{UInt64,UInt128}, m::Int)
    idx = Int[]
    for j in 1:m
        if (mask >> (j - 1)) & 1 == 1
            push!(idx, j)
        end
    end
    return idx
end

# --------------------------------------------
# Evaluate error for a fixed subset mask
# --------------------------------------------
"""
    eval_mask_error(mask::Union{UInt64,UInt128}, X::Matrix{UInt8}, y::Vector{UInt8}) -> Int

Evaluate the majority-vote classification error for the feature subset
specified by `mask`, using a fixed threshold of ⌈k/2⌉ where `k` is the
number of selected features. Returns the number of misclassified samples.
"""
function eval_mask_error(mask::Union{UInt64,UInt128}, X::Matrix{UInt8}, y::Vector{UInt8})
    n, m = size(X)

    counts = zeros(Int16, n)
    k = 0

    for j in 1:m
        if (mask >> (j - 1)) & 1 == 1
            k += 1
            @inbounds for i in 1:n
                counts[i] += X[i, j]
            end
        end
    end

    thresh = (k + 1) >>> 1

    err = 0
    @inbounds for i in 1:n
        pred = counts[i] >= thresh ? 1 : 0
        err += (pred != y[i])
    end

    return err
end

"""
    permutation_test_pvalue(mask, X, y; B=1000, seed=1) -> (observed_error, p_value, perm_errors)

Perform a permutation test to assess the statistical significance of the
classification error achieved by the feature subset given by `mask`.

`B` random label permutations are generated and the classification error is
recomputed for each. The p-value is estimated as
`(count + 1) / (B + 1)`, where `count` is the number of permuted errors
that are ≤ the observed error (when the observed error is > 0).

Returns a named tuple with the observed error, the estimated p-value, and
the vector of permuted errors.
"""
function permutation_test_pvalue(
    mask::Union{UInt64,UInt128},
    X::Matrix{UInt8},
    y::Vector{UInt8};
    B::Int=1000,
    seed::Int=1,
)
    rng = MersenneTwister(seed)

    obs_err = eval_mask_error(mask, X, y)

    n = length(y)
    perm_errs = zeros(Int, B)

    @inbounds for b in 1:B
        y_perm = copy(y)
        shuffle!(rng, y_perm)
        perm_errs[b] = eval_mask_error(mask, X, y_perm)
    end

    count = 0
    @inbounds for b in 1:B
        if perm_errs[b] <= obs_err && obs_err > 0
            count += 1
        end
    end

    p_value = (count + 1) / (B + 1)

    return (observed_error=obs_err, p_value=p_value, perm_errors=perm_errs)
end


"""
    add_bit(mask::UInt128, j::Int) -> UInt128

Return a new mask with bit `j` (1-based) set to 1.
"""
function add_bit(mask::UInt128, j::Int)
    return mask | (UInt128(1) << (j-1))
end

"""
    remove_bit(mask::UInt128, j::Int) -> UInt128

Return a new mask with bit `j` (1-based) cleared to 0.
"""
function remove_bit(mask::UInt128, j::Int)
    return mask & ~(UInt128(1) << (j-1))
end

"""
    has_bit(mask::Union{UInt64,UInt128}, j::Int) -> Bool

Return `true` if bit `j` (1-based) in `mask` is set to 1.
"""
function has_bit(mask::Union{UInt64,UInt128}, j::Int)
    return (mask >> (j-1)) & 1 == 1
end


"""
    compute_error_weighted(counts, k, y; w_pos=1.0, w_neg=1.0) -> (best_err::Float64, best_tau::Int)

Compute the minimum *weighted* classification error over all decision
thresholds τ ∈ {0, …, k}. False negatives are penalised by `w_pos` and
false positives by `w_neg`. Returns the best achievable weighted error and
the corresponding threshold.
"""
function compute_error_weighted(
    counts::Vector{Int16}, k::Int, y::Vector{UInt8}; w_pos::Float64=1.0, w_neg::Float64=1.0
)
    n = length(y)

    hist_pos = zeros(Int, k+1)
    hist_neg = zeros(Int, k+1)

    @inbounds for i in 1:n
        c = counts[i] + 1
        if y[i] == 1
            hist_pos[c] += 1
        else
            hist_neg[c] += 1
        end
    end

    cum_pos = cumsum(hist_pos)
    cum_neg = cumsum(hist_neg)

    total_pos = cum_pos[end]
    total_neg = cum_neg[end]

    best_err = Inf
    best_tau = 0

    @inbounds for τ in 0:k
        fn = τ == 0 ? 0 : cum_pos[τ]
        fp = total_neg - (τ == 0 ? 0 : cum_neg[τ])

        err = w_pos * fn + w_neg * fp

        if err < best_err
            best_err = err
            best_tau = τ
        end
    end

    return best_err, best_tau
end

"""
    compute_class_weights(y::Vector{UInt8}) -> (w_pos::Float64, w_neg::Float64)

Compute inverse-frequency class weights so that the positive and negative
classes contribute equally to the weighted error. Returns
`w_pos = n / (2 * n_pos)` and `w_neg = n / (2 * n_neg)`.
"""
function compute_class_weights(y)
    n = length(y)
    n_pos = sum(y)
    n_neg = n - n_pos

    w_pos = n / (2 * max(n_pos, 1))
    w_neg = n / (2 * max(n_neg, 1))

    return w_pos, w_neg
end

"""
    eval_mask_error_weighted(mask, X, y; w_pos=1.0, w_neg=1.0) -> (err::Float64, tau::Int)

Evaluate the weighted classification error for the feature subset specified
by `mask`. The decision threshold is chosen to minimise the weighted error.
If the mask is empty, the baseline error (minority-class count under the
given weights) is returned.

Returns the best weighted error and the optimal threshold.
"""
function eval_mask_error_weighted(
    mask::Union{UInt64,UInt128},
    X::Matrix{UInt8},
    y::Vector{UInt8};
    w_pos::Float64=1.0,
    w_neg::Float64=1.0,
)
    n, m = size(X)

    if mask == 0
        n_pos = sum(y)
        n_neg = n - n_pos
        return min(w_pos*n_pos, w_neg*n_neg), 0
    end

    counts = zeros(Int16, n)
    k = 0

    @inbounds for j in 1:m
        if (mask >> (j-1)) & 1 == 1
            counts .+= X[:, j]
            k += 1
        end
    end

    err, tau = compute_error_weighted(counts, k, y; w_pos=w_pos, w_neg=w_neg)

    return err, tau
end

"""
    gray_search_weighted(feats::Vector{BitFeature}, y::Vector{UInt8}) -> (best_mask::UInt64, best_err::Float64, best_tau::Int)

Exhaustively enumerate all 2^m feature subsets via Gray code traversal,
optimising the *weighted* classification error jointly with the decision
threshold. Class weights are computed automatically via
[`compute_class_weights`](@ref) to handle imbalanced data. The baseline
error is set to the weighted minority-class count.

Only suitable for m ≤ 64 features.
"""
function gray_search_weighted(feats, y)
    n = length(y)
    m = length(feats)

    counts = zeros(Int16, n)

    w_pos, w_neg = compute_class_weights(y)

    best_err = Inf
    best_mask = UInt64(0)
    best_tau = 0

    prev_gray = UInt64(0)
    k = 0

    total = UInt64(1) << m

    best_err = min(w_pos*sum(y), w_neg*(n-sum(y)))

    @inbounds for t in UInt64(0):(total - 1)
        g = t ⊻ (t >> 1)

        if t > 0
            diff = g ⊻ prev_gray
            j = trailing_zeros(diff) + 1

            if ((g >> (j - 1)) & 1) == 1
                add_feature!(counts, feats[j])
                k += 1
            else
                remove_feature!(counts, feats[j])
                k -= 1
            end
        end

        if k == 0
            prev_gray = g
            continue
        end

        err, tau = compute_error_weighted(counts, k, y; w_pos=w_pos, w_neg=w_neg)

        if err < best_err || (err == best_err && k < count_ones(best_mask))
            best_err = err
            best_mask = g
            best_tau = tau

            if best_err == 0
                break
            end
        end

        prev_gray = g
    end

    return best_mask, best_err, best_tau
end

"""
    sffs_search_weighted(X::Matrix{UInt8}, y::Vector{UInt8}; max_k=typemax(Int), max_iter=1000) -> (best_mask::UInt128, best_err::Float64, best_tau::Int)

Perform Sequential Floating Forward Selection (SFFS) with *weighted*
classification error, suitable for imbalanced datasets. Class weights are
computed automatically via [`compute_class_weights`](@ref). The algorithm
starts from the single best feature and iterates between forward inclusion
and conditional backward exclusion.

Returns the best feature subset mask, the best weighted error, and the
corresponding optimal threshold.
"""
function sffs_search_weighted(
    X::Matrix{UInt8}, y::Vector{UInt8}; max_k::Int=typemax(Int), max_iter::Int=1000
)
    w_pos, w_neg = compute_class_weights(y)

    function best_single()
        m = size(X, 2)
        best_j = 1
        best_err = Inf
        best_tau = 0

        for j in 1:m
            mask = UInt128(1) << (j-1)
            err, tau = eval_mask_error_weighted(mask, X, y; w_pos=w_pos, w_neg=w_neg)

            if err < best_err
                best_err = err
                best_j = j
                best_tau = tau
            end
        end

        return UInt128(1) << (best_j-1), best_err, best_tau
    end

    current_mask, current_err, current_tau = best_single()

    best_mask = current_mask
    best_err = current_err
    best_tau = current_tau

    iter = 0

    while iter < max_iter
        iter += 1
        improved = false

        # Forward
        best_add_err = current_err
        best_add_j = 0

        for j in 1:size(X, 2)
            if !has_bit(current_mask, j)
                new_mask = add_bit(current_mask, j)

                err, tau = eval_mask_error_weighted(
                    new_mask, X, y; w_pos=w_pos, w_neg=w_neg
                )

                if err < best_add_err
                    best_add_err = err
                    best_add_j = j
                    best_tau = tau
                end
            end
        end

        if best_add_j != 0
            current_mask = add_bit(current_mask, best_add_j)
            current_err = best_add_err
            improved = true
        else
            break
        end

        # Backward
        while true
            best_remove_err = current_err
            best_remove_j = 0

            for j in 1:size(X, 2)
                if has_bit(current_mask, j)
                    new_mask = remove_bit(current_mask, j)

                    if new_mask == 0
                        continue
                    end

                    err, tau = eval_mask_error_weighted(
                        new_mask, X, y; w_pos=w_pos, w_neg=w_neg
                    )

                    if err < best_remove_err
                        best_remove_err = err
                        best_remove_j = j
                        best_tau = tau
                    end
                end
            end

            if best_remove_j != 0
                current_mask = remove_bit(current_mask, best_remove_j)
                current_err = best_remove_err
                improved = true
            else
                break
            end
        end

        if current_err < best_err ||
            (current_err == best_err && count_ones(current_mask) < count_ones(best_mask))
            best_mask = current_mask
            best_err = current_err
        end

        if !improved || count_ones(current_mask) >= max_k
            break
        end
    end

    return best_mask, best_err, best_tau
end



# -------------------------
# Full pipeline
# -------------------------
"""
    select_feature_subset(X::Matrix{UInt8}, y::Vector{UInt8}) -> (indices::Vector{Int}, error::Union{Int,Float64}, p_value::Float64, tau::Int)

High-level entry point that selects the optimal subset of binary features
for majority-vote classification.

## Algorithm selection
- `m ≤ 20`: exact Gray code enumeration with weighted error
  ([`gray_search_weighted`](@ref)).
- `20 < m ≤ 128`: Sequential Floating Forward Selection with weighted error
  ([`sffs_search_weighted`](@ref), `max_k = 15`).
- `m > 128`: raises an error — use a random-forest or Lasso-based method
  instead.

After the best subset is found, a permutation test ([`permutation_test_pvalue`](@ref)
with `B = 1000` permutations) is performed to estimate statistical
significance.

## Returns
- `indices`: 1-based feature indices of the selected subset.
- `error`: classification error of the selected subset.
- `p_value`: permutation-test p-value.
- `tau`: optimal decision threshold.
"""
function select_feature_subset(X::Matrix{UInt8}, y::Vector{UInt8})
    n, m = size(X)

    if m <= 20
        # Exact Gray code enumeration
        feats = build_features(X)
        feats = sort_features!(feats, y)
        mask, err, tau = gray_search_weighted(feats, y)
    elseif m <= 128
        # Sequential Floating Forward Selection
        mask, err, tau = sffs_search_weighted(X, y; max_k=15)
    else
        error("Too many features ($m > 128). Use RFMethod or LassoMethod instead.")
    end

    test = permutation_test_pvalue(mask, X, y; B=1000)

    return mask_to_indices(mask, m), err, test.p_value, tau
end
