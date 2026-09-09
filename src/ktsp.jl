using Statistics, DataFrames

"""
    fit_ktsp(data, labels, gene_names, cfg; k_max=9) -> KTSPModel

Train a k-Top Scoring Pairs model. Greedily selects up to `k_max` disjoint
gene pairs ranked by |p1 − p0|, enforcing odd k to avoid ties.
"""
function fit_ktsp(
    data::Matrix{T}, labels::AbstractVector, gene_names::Vector, cfg::REOConfig; k_max=9
) where {T<:Real}
    selected_genes = filter_genes(data, labels, gene_names, cfg)
    data = data[selected_genes, :]
    gene_names = gene_names[selected_genes]
    n_genes, n_samples = size(data)

    idx0 = findall(==(0), labels)
    idx1 = findall(==(1), labels)
    n0, n1 = length(idx0), length(idx1)

    # Score all pairs
    all_scores = []
    for i in 1:(n_genes - 1)
        for j in (i + 1):n_genes
            p0 = sum(data[i, idx0] .< data[j, idx0]) / n0
            p1 = sum(data[i, idx1] .< data[j, idx1]) / n1
            Δ = abs(p1 - p0)
            if Δ > 0
                push!(all_scores, (i, j, Δ, p1 > p0))
            end
        end
    end

    sort!(all_scores; by=x -> x[3], rev=true)

    # Greedy disjoint selection
    selected_pairs = Tuple{Int,Int}[]
    selected_scores = Float64[]
    directions = Bool[]
    used_genes = Set{Int}()

    for (i, j, Δ, dir) in all_scores
        length(selected_pairs) >= k_max && break
        if !(i in used_genes) && !(j in used_genes)
            push!(selected_pairs, (i, j))
            push!(selected_scores, Δ)
            push!(directions, dir)
            push!(used_genes, i)
            push!(used_genes, j)
        end
    end

    # Enforce odd k to avoid ties
    k_final = length(selected_pairs)
    if k_final % 2 == 0 && k_final > 0
        pop!(selected_pairs);
        pop!(selected_scores);
        pop!(directions)
        k_final -= 1
    end

    return KTSPModel(
        selected_pairs,
        [(gene_names[p[1]], gene_names[p[2]]) for p in selected_pairs],
        selected_scores,
        directions,
        k_final,
    )
end

"""
    predict_ktsp(model, new_data, gene_names) -> BitVector

Predict class labels using majority vote over a trained k-TSP model.
"""
function predict_ktsp(
    model::KTSPModel, new_data::Matrix{T}, gene_names::Vector
) where {T<:Real}
    gene_to_row = Dict(gene => i for (i, gene) in enumerate(gene_names))
    n_samples = size(new_data, 2)
    votes = zeros(Int, n_samples)

    for (idx, (gene_i, gene_j)) in enumerate(model.gene_names)
        i = gene_to_row[gene_i]
        j = gene_to_row[gene_j]
        is_less = new_data[i, :] .< new_data[j, :]
        if model.p_directions[idx]
            votes .+= is_less
        else
            votes .+= .!is_less
        end
    end

    return (votes .> (model.k / 2))
end
