"""
Generate simulated test data with two informative gene pairs.
"""
function generate_test_data(n_genes=1000, n_samples=200)
    data = randn(n_genes, n_samples)
    labels = vcat(ones(Int, n_samples ÷ 2), zeros(Int, n_samples ÷ 2))
    # Inject a strong G1 > G2 signal for the positive class
    data[1, labels .== 1] .+= 2.0
    data[2, labels .== 1] .-= 2.0
    genes = ["Gene_$i" for i in 1:n_genes]
    return data, labels, genes
end
