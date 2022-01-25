# SepdQuantile

[![Build Status](https://github.com/lukketotte/SepdQuantile.jl/workflows/CI/badge.svg)](https://github.com/lukketotte/SepdQuantile.jl/actions)
[![Coverage](https://codecov.io/gh/lukketotte/SepdQuantile.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/lukketotte/EpdTest.jl)

**SepdQuantile.jl** is a Julia library created for the paper *Quantile regression based on the skewed exponential power distribution*.

## Installation
Through the `pkg` REPL mode by typing
```
] add "https://github.com/lukketotte/SepdQuantile.jl"
```


## Recreate results
To recreate Fig. 4,5, and 6 for sample size n = 1000, the simulation is carried out as below.
```julia
using Distributed, SharedArrays
@everywhere using SepdQuantile, LinearAlgebra, StatsBase, QuantileRegressions, DataFrames
n = 1000;
x = rand(Normal(), n);
X = hcat(ones(n), x)

p = [1.5, 2., 2.5]
skew = [0.1, 0.9, 0.5]
quant = [0.1, 0.5, 0.9]

settings = DataFrame(p = repeat(p, inner = length(skew)) |> x -> repeat(x, inner = length(quant)),
    skew = repeat(skew, length(p)) |> x -> repeat(x, inner = length(quant)),
    tau = repeat(quant, length(skew) * length(p)), old = 0,  sdOld = 0, bayes = 0,
    sdBayes = 0, freq = 0, sdFreq = 0)

cols = names(settings)
settings = SharedArray(Matrix(settings))
reps = 100

control =  Dict(:tol => 1e-3, :max_iter => 1000, :max_upd => 0.3,
  :is_se => false, :est_beta => true, :est_sigma => true,
  :est_p => true, :est_tau => true, :log => false, :verbose => false)

@sync @distributed for i ∈ 1:size(settings, 1)
    println(i)
    p, skew, τ = settings[i, 1:3]
    old, bayes, freq = [zeros(reps) for i in 1:3]
    for j ∈ 1:reps
        y = 2.1 .+ 0.5 .* x + rand(Aepd(0, 1, p, skew), n);

        # Bayesian
        par = Sampler(y, X, skew, 10000, 5, 2500);
        β, θ, σ, α = mcmc(par, 0.8, .25, 1.5, 1, 2, 0.5, [2.1, 0.5], verbose = false);
        μ = X * median(β, dims = 1)' |> x -> reshape(x, size(x, 1))

        # Freq
        control[:est_sigma], control[:est_tau], control[:est_p] = (true, true, true)
        res = quantfreq(y, X, control)
        μf = X * res[:beta] |> x -> reshape(x, size(x, 1))

        # Compute τ converted
        b = DataFrame(hcat(par.y, par.X), :auto) |> x -> qreg(@formula(x1 ~  x3), x, τ) |> coef
        q = X * b

        taubayes = [quantconvert(q[k], median(θ), median(α), μ[k], median(σ)) for k in 1:length(par.y)] |> mean
        taufreq  = [quantconvert(q[k], res[:p], res[:tau], μf[k], res[:sigma]) for k in 1:length(y)] |> mean

        par.α = taubayes
        βres = mcmc(par, 1.3, median(θ), median(σ), b, verbose = false)
        μ = X * median(βres, dims = 1)' |> x -> reshape(x, size(x, 1))
        bayes[j] = [par.y[k] <= μ[k]  for k in 1:length(par.y)] |> mean

        par.α = τ
        par.πθ = "uniform"
        βt, _, _ = mcmc(par, .6, .6, 1.2, 2, b, verbose = false)
        μ = X * median(βt, dims = 1)' |> x -> reshape(x, size(x, 1))
        old[j] = [par.y[k] <= μ[k]  for k in 1:length(par.y)] |> mean

        control[:est_sigma], control[:est_tau], control[:est_p] = (false, false, false)
        res = quantfreq(y, X, control, res[:sigma], res[:p], taufreq)
        freq[j] = mean(y .<= X*res[:beta])

    end
    settings[i, 4] = mean(old)
    settings[i, 5] = √var(old)
    settings[i, 6] = mean(bayes)
    settings[i, 7] = √var(bayes)
    settings[i, 8] = mean(freq)
    settings[i, 9] = √var(freq)
end

plt_dat = DataFrame(Tables.table(settings)) |> x -> rename!(x, cols)
```
