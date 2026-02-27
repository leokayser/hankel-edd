# Start Julia with multiple threads, e.g., julia --threads 6
# Check number of threads: Threads.nthreads()

using HomotopyContinuation
include("auxiliaryFunctions.jl")

y = complex(vec([0.6,-0.5,-0.1,0.6,-1.0,-0.3,-1.1,-0.6,0.9,-1.4]))
N = length(y)
r = 2
we = "bomb"

### Solve using FONC
sys = Walsh.construct_fonc(y, r; weights=we)
res = solve(sys, only_non_zero=false, threading=true, compile=true)
tab = Walsh.analyze_solutions(y, r, res; system="lagrangian", weights=we, only_real=true)

### Solve using Walsh
sys = Walsh.construct_walsh(y, r; weights=we)
res = solve(sys, only_non_zero=false, threading=true, compile=true)
tab = Walsh.analyze_solutions(y, r, res; system="walsh", weights=we, only_real=true)

### Create contourplot
using Plots
w = Walsh.get_weights(we, N, r); Λ = diagm(w);
f(a0,a1) = norm(sqrt.(Λ)*(y-Walsh.compute_optimal_y_hat(y,ComplexF64.([a0,a1,1]); weights=we)),2)^2;
a1 = range(-10, 6, length=100);
a0 = range(-10, 5, length=100);
z = zeros(length(a1),length(a0));
for i in eachindex(a1)
    for j in eachindex(a0)
        z[i,j] = f(a1[i], a0[j]);
    end
end
contour(a1, a0, z)