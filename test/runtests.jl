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
        L = nlevels + 1
        trans = MODWT(wt, nlevels)
    end
    return esc(ex)
end

@testset "Utilities" begin
    @setupvars
    X = randn(n, mx)
    # MODWT
    Xw = modwt_matrix(X, trans)
    X1 = imodwt_matrix(Xw, trans)
    @test size(Xw) == (n, nlevels+1, mx)
    @test size(X1) == size(X)
    @test all(X .≈ X1)

    # DWT
    trans = DWT(wt, nlevels)
    Xw = dwt_matrix(X, trans)
    X1 = idwt_matrix(Xw, trans)
    @test size(Xw) == (n, mx)
    @test size(X1) == size(X)
    @test all(X .≈ X1)
end

# vector X and Y
@testset "Univariate timeseries" begin
    @setupvars
    X = randn(n)
    Y = randn(n)
    res1 = wavelm(X, Y, trans)
    @test length(res1.coefficients) == L
    @test all(size(B) == (1, 1) for B  in res1.coefficients)
    Y_pred = predict(res1)
    Y_pred = predict(res1, randn(length(X)))
    Y_pred = predict(res1, randn(length(X) + 1))
end

# matrix X and Y
@testset "Vector timeseries" begin
    @setupvars
    X = randn(n, mx)
    Y = randn(n, my)
    res1 = wavelm(X, Y, trans)
    @test length(res1.coefficients) == L
    @test all(size(B) == (mx, my) for B  in res1.coefficients)
    @test ncoef(res1) == mx * my * (L)
    @test dof(res1) == n - mx * my * (L) - 1
    @test size(regression_stderr(res1)) == (my, L)
    Y_pred = predict(res1)
    Y_pred = predict(res1, randn(size(X)))
    Y_pred = predict(res1, randn(size(X, 1) + 1, size(X, 2)))
    SE = coef_stderr(res1)
    BB = coef(res1)
    BBshrunk = coef_shrunk(res1)
    @test all([B != Bs for (B, Bs) in zip(BB, BBshrunk)])
    res1shrunk = shrink(res1)
    @test length(SE) == length(BB) == L
    @test all([size(SE[i]) == size(BB[i]) for i in 1:L])
    @test all([all(se .>= 0) for se in SE])
    trans = DWT(wt, nlevels)
    res2 = wavelm(X, Y, trans)
end
