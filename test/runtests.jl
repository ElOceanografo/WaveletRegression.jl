using WaveletRegression
using Wavelets
using Statistics
using Test


macro setupvars()
    ex = quote
        n = 128
        mx = 5
        my = 3
        wt = wavelet(WT.db2)
        nlevels = 4
    end
    return esc(ex)
end

@testset "Utilities" begin
    @setupvars
    X = randn(n, mx)
    Xw = modwt_matrix(X, wt, nlevels)
    X1 = imodwt_matrix(Xw, wt)
    @test size(Xw) == (n, nlevels+1, mx)
    @test size(X1) == size(X)
    @test all(X .≈ X1)
end

# vector X and Y
@testset "Univariate timeseries" begin
    @setupvars
    X = randn(n)
    Y = randn(n)
    res1 = wavelm(X, Y, wt, nlevels)
    @test length(res1.coefficients) == nlevels + 1
    @test all(size(B) == (1, 1) for B  in res1.coefficients)
end

# matrix X and Y
@testset "Vector timeseries" begin
    @setupvars
    X = randn(n, mx)
    Y = randn(n, my)
    res1 = wavelm(X, Y, wt, nlevels)
    @test length(res1.coefficients) == nlevels + 1
    @test all(size(B) == (mx, my) for B  in res1.coefficients)
end

# @setupvars
# coefficients = [randn(mx, my) for i in 1:nlevels+1]
# Xw = modwt_matrix(X, wt, nlevels)
# @test all(imodwt_matrix(Xw, wt) .≈ X)
# @test all(modwt_matrix(X, wt, nlevels) .≈ Xw)
#
# Yw_true = cat([reshape(Xw[:, j, :] * coefficients[j], (n, 1, my))
#     for j in 1:nlevels+1]..., dims=2)
# Y_true = imodwt_matrix(Yw_true, wt)
# Yw1 = modwt_matrix(Y_true, wt, nlevels)
# mean(abs2.(Yw1 .- Yw_true))
#
#
# fit = wavelm(X, Y_true, wt, nlevels)
# predict(fit)
#
#
# plot(Y_true, color=:black, layout=(3,1), legend=false)
# plot!(predict(fit), color=:red, layout=(3,1))
#
# BB = [Xw[:, j, :] \ Yw1[:, j, :] for j in 1:(nlevels+1)]
# Yw_pred = cat([reshape(Xw[:, j, :] * BB[j], (n, 1, my))
#     for j in 1:nlevels+1]..., dims=2)
# Y_pred = imodwt_matrix(Yw_pred, wt)
# plot!(Y_pred, color=:blue, layout=(3,1))
