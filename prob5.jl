using LinearAlgebra, PyPlot, Statistics, Printf

function projector(b::Vector)
    return I - b*b'
end

function ODE_rhs_operator(A::Matrix, b::Vector, t::Number)
    Qb = projector(b)
    t = real(t)
    return (1.0 - t)*Qb + t*A*Qb*A
end

function ODE_rhs(u::Vector, t::Number, A::Matrix, b::Vector,
    S::AbstractFloat)

    H = ODE_rhs_operator(A, b, t)
    return -im*S*H*u
end

function autonomized_rhs(xi::Vector, A::Matrix, b::Vector, S::AbstractFloat)
    u = xi[1:end-1]
    t = xi[end]
    f = ODE_rhs(u, t, A, b, S)
    return vcat(f, 1.0)
end

function rk4Stages(solution::Vector, A::Matrix, b::Vector, S::AbstractFloat,
    dt::AbstractFloat)

    k1 = autonomized_rhs(solution, A, b, S)
    k2 = autonomized_rhs(solution+dt/2*k1, A, b, S)
    k3 = autonomized_rhs(solution+dt/2*k2, A, b, S)
    k4 = autonomized_rhs(solution+dt*k3, A, b, S)
    return k1, k2, k3, k4
end

function stepRK4(solution::Vector, A::Matrix, b::Vector, S::AbstractFloat,
    dt::AbstractFloat)

    k1, k2, k3, k4 = rk4Stages(solution, A, b, S, dt)
    return solution + dt/6*(k1+2k2+2k3+k4)
end

function runRK4Iterations(xi0::Vector, A, b, S, dt, stop_time)

    xi = copy(xi0)
    current_time = 0.0
    @assert current_time < stop_time
    while !(isapprox(current_time - stop_time, 0.0, atol = dt/2))
        current_time += dt
        xi = stepRK4(xi, A, b, S, dt)
        println("Current time = ", current_time)
    end
    return xi
end

function GL2Stages(stages::Vector, solution::Vector,
    A::Matrix, b::Vector, S::AbstractFloat, step_size::AbstractFloat)

    d = length(solution)
    k1 = stages[1:d]
    k2 = stages[d+1:end]
    k1_next = autonomized_rhs(solution+step_size*(1/4*k1+(1/4-sqrt(3)/6)*k2), A, b, S)
    k2_next = autonomized_rhs(solution+step_size*((1/4+sqrt(3)/6)*k1+1/4*k2), A, b, S)
    return vcat(k1_next, k2_next)
end

function GL2StagesNonlinear(stages::Vector, solution::Vector,
    A::Matrix, b::Vector, S::AbstractFloat, step_size::AbstractFloat)

    return stages - GL2Stages(stages, solution, A, b, S, step_size)
end


function broydenLagrangeMultiplier(Bprev::Matrix, y::AbstractArray, s::AbstractArray)
    return (Bprev*y - s)/dot(y,y)
end

function updateBroydenMatrix(Bprev::Matrix, y::AbstractArray, s::AbstractArray)
    lambda = broydenLagrangeMultiplier(Bprev, y, s)
    return Bprev - lambda*y'
end


function initialBroydenMatrix(alpha, ndofs)
    return alpha*Matrix((1.0+0.0im)*I, ndofs, ndofs)
end

function stepBroydenUpdate(Kprev::Vector, Bprev::Matrix, Gprev::Vector,
    solution::Vector, A::Matrix, b::Vector, S::AbstractFloat, dt::AbstractFloat)

    Knext = Kprev - Bprev*Gprev
    s = Knext - Kprev
    Gnext = GL2StagesNonlinear(Knext, solution, A, b, S, dt)
    y = Gnext - Gprev
    lambda = broydenLagrangeMultiplier(Bprev, y, s)
    Bnext = Bprev - lambda*y'
    return Knext, Bnext, Gnext
end

function runBroydenIterations(K0::Vector, B::Matrix, xi::Vector,
    A, b, S, dt; tol = 1e-11, maxiter = 200)

    K = copy(K0)
    G = GL2StagesNonlinear(K, xi, A, b, S, dt)
    err = norm(G)
    count = 0
    while err > tol && count < maxiter
        count += 1
        K, B, G = stepBroydenUpdate(K, B, G, xi, A, b, S, dt)
        err = norm(G)
    end
    println("\t # Broyden iterations = ", count)
    println("\t Nonlinear function value = ", err)
    if count == maxiter || isnan(err)
        error("Broyden iterations did not converge after $maxiter iterations")
    end
    return K, B
end

function stepGL2(xi::Vector, K::Vector, dt::AbstractFloat)
    d = length(xi)
    k1 = K[1:d]
    k2 = K[d+1:end]
    return xi + dt/2*(k1 + k2)
end

function runGL2Iterations(xi0::Vector, A, b, S, dt, stop_time;
    broyden_alpha = 0.0001)

    xi = copy(xi0)
    current_time = 0.0
    K = vcat(xi0, xi0)
    B = initialBroydenMatrix(broyden_alpha, length(K))
    while current_time < stop_time && !(isapprox(current_time - stop_time, 0.0, atol = 1e-10))
        current_time += dt
        K, B = runBroydenIterations(K, B, xi, A, b, S, dt)
        xi = stepGL2(xi, K, dt)
        println("Current time = ", current_time)
    end
    return xi
end

function analyticalSolution(A, b)
    x = A\b
    return x/norm(x)
end

function getError(u, x)
    return norm(u*u' - x*x')
end

function getErrorVsStepSize(scheme::Function, A, b, S)
    stop_time = 1.0
    exponent = 1:5
    step_size = [10.0^(-k) for k in exponent]
    err = zeros(length(step_size))
    analytical_solution = analyticalSolution(A,b)
    u0 = b/norm(b)
    xi0 = complex(vcat(u0, 0.0))
    for (idx,dt) in enumerate(step_size)
        xi = scheme(xi0, A, b, S, dt, stop_time)
        u = xi[1:end-1]
        err[idx] = getError(u,analytical_solution)
    end
    return step_size, err
end

function getErrorVsS(A, b, S)
    Slist = [1e3, 2e3, 5e3, 1e4]
    stop_time = 1.0
    # dt_list = [1e-2, 1e-2, 1e-2, 1e-3]
    dt = 1e-3
    err = zeros(length(Slist))
    analytical_solution = analyticalSolution(A,b)
    u0 = b/norm(b)
    xi0 = complex(vcat(u0, 0.0))
    for (idx,S) in enumerate(Slist)
        xi = runGL2Iterations(xi0, A, b, S, dt, stop_time)
        u = xi[1:end-1]
        err[idx] = getError(u,analytical_solution)
    end
    return Slist, err
end

function plotError(x, err; title = "", xlabel = "",
    slope = false)

    fig, ax = PyPlot.subplots()
    ax.loglog(x, err, "-o", linewidth = 2)
    ax.grid()
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Error")
    ax.set_title(title)
    if slope
        s = mean(diff(log.(err)) ./ diff(log.(x)))
        annotation = @sprintf "Mean slope = %1.2f" s
        ax.annotate(annotation, (0.3, 0.4), xycoords = "axes fraction")
    end
    ax.set_ylim([1e-3, 1])
    return fig
end

A = 1.0/6.0*[4.0    1.0     1.0
             1.0    3.0     1.0
             1.0    1.0     2.0]
b = 1.0/sqrt(3.0)*[1.0, 1.0, 1.0]
S = 1e4
k = 2
dt = 10.0^(-k)
stop_time = 1.0
broyden_alpha = 0.0001

# step_size, err = getErrorVsStepSize(runGL2Iterations, A, b, S)
# fig = plotErrorVsStepSize(step_size, err, title = "Error vs step size for RK4")

Slist, err = getErrorVsS(A, b, S)
plotError(Slist, err, xlabel = "S", slope = true)

# x = A\b
# x = x/norm(x)
# u0 = b/norm(b)
# xi0 = complex(vcat(u0, 0.0))
# xi = runGL2Iterations(xi0, A, b, S, dt, stop_time, broyden_alpha = broyden_alpha)
# xi = runRK4Iterations(xi0, A, b, S, dt, stop_time)
# u = xi[1:end-1]
# println("Error = ", norm(u*u' - x*x'))
