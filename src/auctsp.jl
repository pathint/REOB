using Statistics, DataFrames

"""
    fit_auctsp(data, labels, gene_names, cfg; k_max=9) -> AUCTSPModel

Train an AUC-based TSP model. For each gene pair, computes AUC in both
ordering directions and greedily selects up to `k_max` disjoint pairs.
"""
function fit_auctsp(
    data::Matrix{T}, labels::AbstractVector, gene_names::Vector, cfg::REOConfig; k_max=9
) where {T<:Real}
    selected_genes = filter_genes(data, labels, gene_names, cfg)
    data = data[selected_genes, :]
    gene_names = gene_names[selected_genes]

    n_genes, n_samples = size(data)
    idx0 = findall(==(0), labels)
    idx1 = findall(==(1), labels)
    n0, n1 = length(idx0), length(idx1)

    all_pairs_stats = []

    for i in 1:(n_genes - 1)
        for j in (i + 1):n_genes
            p_ij_1 = sum(data[i, idx1] .< data[j, idx1]) / n1
            p_ij_0 = sum(data[i, idx0] .< data[j, idx0]) / n0

            # AUC for both directions
            auc_a = (p_ij_1 + (1 - p_ij_0)) / 2  # Xi < Xj → Class 1
            auc_b = ((1 - p_ij_1) + p_ij_0) / 2   # Xi > Xj → Class 1

            best_auc = max(auc_a, auc_b)
            direction = auc_a >= auc_b ? 1 : -1

            secondary = abs(
                mean(data[i, idx1] .- data[j, idx1]) - mean(data[i, idx0] .- data[j, idx0])
            )

            push!(all_pairs_stats, (i, j, best_auc, secondary, direction))
        end
    end

    sort!(all_pairs_stats; by=x -> (x[3], x[4]), rev=true)

    # Greedy disjoint selection
    selected_pairs = []
    used_genes = Set{Int}()

    for (i, j, auc_val, _, dir) in all_pairs_stats
        length(selected_pairs) >= k_max && break
        if !(i in used_genes) && !(j in used_genes)
            push!(selected_pairs, (i, j, auc_val, dir))
            push!(used_genes, i)
            push!(used_genes, j)
        end
    end

    k_final = length(selected_pairs)

    return AUCTSPModel(
        [(p[1], p[2]) for p in selected_pairs],
        [(gene_names[p[1]], gene_names[p[2]]) for p in selected_pairs],
        [p[3] for p in selected_pairs],
        [p[4] for p in selected_pairs],
        k_final,
    )
end

"""
    predict_auctsp(model, new_data, gene_names) -> BitVector

Predict class labels using majority vote over a trained AUC-TSP model.
"""
function predict_auctsp(
    model::AUCTSPModel, new_data::Matrix{T}, gene_names::Vector
) where {T<:Real}
    gene_to_row = Dict(gene => i for (i, gene) in enumerate(gene_names))
    n_samples = size(new_data, 2)
    votes = zeros(Float64, n_samples)

    for (idx, (name_i, name_j)) in enumerate(model.gene_names)
        i = gene_to_row[name_i]
        j = gene_to_row[name_j]
        if model.directions[idx] == 1
            votes .+= (new_data[i, :] .< new_data[j, :])
        else
            votes .+= (new_data[i, :] .> new_data[j, :])
        end
    end

    return (votes .> (model.k / 2))
end
