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
        trans = MODWT(wt, nlevels)
    end
    return esc(ex)
end

@testset "Utilities" begin
    @setupvars
    X = randn(n, mx)
    Xw = modwt_matrix(X, trans)
    X1 = imodwt_matrix(Xw, trans)
    @test size(Xw) == (n, nlevels+1, mx)
    @test size(X1) == size(X)
    @test all(X .≈ X1)
end

# vector X and Y
@testset "Univariate timeseries" begin
    @setupvars
    X = randn(n)
    Y = randn(n)
    res1 = wavelm(X, Y, trans)
    @test length(res1.coefficients) == nlevels + 1
    @test all(size(B) == (1, 1) for B  in res1.coefficients)
    Y_pred = predict(res1)
    Y_pred = predict(res1, randn(size(X)))
end

# matrix X and Y
@testset "Vector timeseries" begin
    @setupvars
    X = randn(n, mx)
    Y = randn(n, my)
    res1 = wavelm(X, Y, trans)
    @test length(res1.coefficients) == nlevels + 1
    @test all(size(B) == (mx, my) for B  in res1.coefficients)
    Y_pred = predict(res1)
    Y_pred = predict(res1, randn(size(X)))
end
