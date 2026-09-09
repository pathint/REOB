using REOBiomarker
using Test
using Statistics

# Generate simulated data: 1000 genes, 200 samples
# Gene_1 and Gene_2 are designed to have strong discriminative power
data, labels, genes = generate_test_data(1000, 200)

@testset "REOBiomarker full pipeline" begin
    @testset "Low-level filter functions (filters.jl)" begin
        keep_low = REOBiomarker.filter_low_rank_genes(data, 0.1)
        @test length(keep_low) <= 1000
        @test length(keep_low) > 0

        keep_diff = REOBiomarker.filter_diff_rank_genes(data, labels, keep_low, top_n=100)
        @test length(keep_diff) <= 100

        pairs, pvals = REOBiomarker.get_top_pairs_parallel_fisher(
            data, labels, keep_diff, n_top=50
        )
        @test length(pairs) > 0
        @test pvals[1] <= pvals[end]
    end

    @testset "VotingMethod" begin
        cfg_vote = REOConfig(
            method=VotingMethod,
            top_diff_n=500,
            bqc_threshold=1.0,
            p0_threshold=0.05,
            verbose=false,
        )

        model_vote = fit_reo(data, labels, genes, cfg_vote)
        @test model_vote.config.method == VotingMethod
        @test length(model_vote.final_pairs) > 0
        @test all(model_vote.weights .> 0)
        @test sum(model_vote.weights) ≈ 1.0 atol=0.01

        res = evaluate_reo(model_vote, data, genes, labels)
        @test res.acc >= 0.5
        @test 0.0 <= res.auc <= 1.0
        @test -1.0 <= res.mcc <= 1.0

        pred = predict_reo(model_vote, data, genes)
        @test length(pred.probs) == size(data, 2)
        @test length(pred.preds) == size(data, 2)
    end

    @testset "RFMethod" begin
        cfg_rf = REOConfig(
            method=RFMethod,
            target_n=5,
            ss_iterations=100,
            bqc_threshold=1.0,
            p0_threshold=0.05,
            verbose=false,
        )

        model_rf = fit_reo(data, labels, genes, cfg_rf)
        @test model_rf.config.method == RFMethod
        @test length(model_rf.final_pairs) <= 5
        @test all(model_rf.weights .> 0)

        res = evaluate_reo(model_rf, data, genes, labels)
        @test res.acc >= 0.5
        @test 0.0 <= res.auc <= 1.0
        @test -1.0 <= res.mcc <= 1.0
    end

    @testset "LassoMethod" begin
        cfg_lasso = REOConfig(
            method=LassoMethod,
            target_n=5,
            bqc_threshold=1.0,
            p0_threshold=0.05,
            verbose=false,
        )

        model_lasso = fit_reo(data, labels, genes, cfg_lasso)
        @test model_lasso.config.method == LassoMethod

        res = evaluate_reo(model_lasso, data, genes, labels)
        @test res.acc >= 0.5
    end

    @testset "Permutation test" begin
        cfg = REOConfig(
            method=RFMethod,
            target_n=3,
            ss_iterations=50,
            bqc_threshold=1.0,
            p0_threshold=0.05,
            verbose=false,
        )
        model = fit_reo(data, labels, genes, cfg)

		perm_res = run_permutation_test(data, labels, genes, model, 
										  cfg, n_permutations=10)
        @test haskey(perm_res, :p_value)
        @test 0.0 <= perm_res.p_value <= 1.0
    end

    @testset "TSP baseline" begin
        cfg = REOConfig(low_rank_q=0.0, top_diff_n=500)
        tsp = fit_tsp(data, labels, genes, cfg)
        @test tsp.gene_names[1] != tsp.gene_names[2]
        @test 0.0 <= tsp.score <= 1.0

        preds = predict_tsp(tsp, data, genes)
        @test length(preds) == size(data, 2)
    end

    @testset "k-TSP baseline" begin
        cfg = REOConfig(low_rank_q=0.0, top_diff_n=200)
        ktsp = fit_ktsp(data, labels, genes, cfg; k_max=5)
        @test ktsp.k <= 5
        @test ktsp.k % 2 == 1  # enforced odd

        preds = predict_ktsp(ktsp, data, genes)
        @test length(preds) == size(data, 2)
    end

    @testset "AUC-TSP baseline" begin
        cfg = REOConfig(low_rank_q=0.0, top_diff_n=200)
        auctsp = fit_auctsp(data, labels, genes, cfg; k_max=5)
        @test length(auctsp.pairs) <= 5

        preds = predict_auctsp(auctsp, data, genes)
        @test length(preds) == size(data, 2)
    end

    @testset "Error handling" begin
        cfg = REOConfig(target_n=3, bqc_threshold=1.0, p0_threshold=0.05, verbose=false)
        model = fit_reo(data, labels, genes, cfg)

        wrong_genes = ["Wrong_$i" for i in 1:1000]
        @test_throws ErrorException predict_reo(model, data, wrong_genes)
    end
end
