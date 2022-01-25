module SepdQuantile

using Distributions, LinearAlgebra, StatsBase, SpecialFunctions, ProgressMeter, ForwardDiff
using DataFrames

import Distributions: pdf, _logpdf, sampler, _rand!,
    logpdf, @check_args
import Base: rand

export mcmc, Sampler, acceptance, Aepd


end # module
