using Statistics

"""
    fit_tsp(data, labels, gene_names, cfg) -> TSPModel

Train a Top Scoring Pair model. Finds the single gene pair whose class-wise
ordering frequency difference |p1 − p0| is maximal.
"""
function fit_tsp(
    data::Matrix{T}, labels::AbstractVector, gene_names::Vector, cfg::REOConfig
) where {T<:Real}
    # Pre-filter genes
    selected_genes = filter_genes(data, labels, gene_names, cfg)
    data = data[selected_genes, :]
    gene_names = gene_names[selected_genes]
    n_genes, n_samples = size(data)

    idx0 = findall(==(0), labels)
    idx1 = findall(==(1), labels)
    n0, n1 = length(idx0), length(idx1)

    best_score = -1.0
    best_secondary = -1.0
    best_pair = (0, 0)

    # Pre-compute per-sample ranks for tie-breaking
    ranks = zeros(Float64, n_genes, n_samples)
    for j in 1:n_samples
        ranks[:, j] .= tiedrank(data[:, j])
    end

    # Exhaustive O(G^2 * S) scan
    for i in 1:(n_genes - 1)
        for j in (i + 1):n_genes
            k0 = sum(data[i, idx0] .< data[j, idx0])
            k1 = sum(data[i, idx1] .< data[j, idx1])
            p0 = k0 / n0
            p1 = k1 / n1
            Δ = abs(p1 - p0)

            if Δ > best_score
                best_score = Δ
                best_pair = (i, j)
                best_secondary = abs(
                    mean(ranks[i, idx1] .- ranks[j, idx1]) -
                    mean(ranks[i, idx0] .- ranks[j, idx0]),
                )
            elseif Δ == best_score && Δ > 0
                secondary = abs(
                    mean(ranks[i, idx1] .- ranks[j, idx1]) -
                    mean(ranks[i, idx0] .- ranks[j, idx0]),
                )
                if secondary > best_secondary
                    best_secondary = secondary
                    best_pair = (i, j)
                end
            end
        end
    end

    i, j = best_pair
    p0_final = mean(data[i, idx0] .< data[j, idx0])
    p1_final = mean(data[i, idx1] .< data[j, idx1])

    return TSPModel(i, j, (gene_names[i], gene_names[j]), best_score, p0_final, p1_final)
end

"""
    predict_tsp(model, new_data, gene_names) -> BitVector

Predict class labels using a trained TSP model.
"""
function predict_tsp(
    model::TSPModel, new_data::Matrix{T}, gene_names::Vector
) where {T<:Real}
    gene_to_row = Dict(gene => i for (i, gene) in enumerate(gene_names))
    name1, name2 = model.gene_names
    gene_i = gene_to_row[name1]
    gene_j = gene_to_row[name2]

    is_less = new_data[gene_i, :] .< new_data[gene_j, :]

    if model.p1 > model.p0
        return is_less
    else
        return .!is_less
    end
end
