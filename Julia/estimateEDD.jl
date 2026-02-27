# Start Julia with multiple threads, e.g., julia --threads 6
# Check number of threads: Threads.nthreads()

using HomotopyContinuation
using ToeplitzMatrices
using LinearAlgebra
include("auxiliaryFunctions.jl")

max_N = 10
max_r = 2
weights="bomb"

num_real = zeros(Int, max_N, max_r);
num_complex = zeros(Int, max_N, max_r);
num_regular = zeros(Int, max_N, max_r);

for N in range(1, max_N)
    y = complex(vec(randn(N, 1)))
    for r in range(1, max_r)

        # Skip this iteration if N > 2n is not satisfied
        if (2 * r >= N)
            continue
        end

        sys = Walsh.construct_walsh(y, r; weights=weights)
        vars = variables(sys)
        res = solve(sys, only_non_zero=false, threading=true, compile=true)
        num_complex[N, r] = nresults(
            res;
            only_real=false,
            real_tol=1e-6,
            only_nonsingular=false,
            only_singular=false,
            only_finite=true,
            multiple_results=false,
        )
        num_real[N, r] = nresults(
            res;
            only_real=true,
            real_tol=1e-6,
            only_nonsingular=false,
            only_singular=false,
            only_finite=true,
            multiple_results=false,
        )

        # Compute Y_hat and check rank:
        for sol in solutions(res)
            y_hat = Walsh.compute_optimal_y_hat(y, sol[1:r+1]; weights=weights)
            Y_hat = Hankel(y_hat, (N - r, r + 1))
            rr = rank(Y_hat, rtol=0.000001)
            if rr == r
                num_regular[N, r] += 1
            elseif rr > r
                print("---- WARNING: full-rank Y_hat (r > n) ----")
            end
        end
    end
end