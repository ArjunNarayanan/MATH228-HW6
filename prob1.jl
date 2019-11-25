using LinearAlgebra, SparseArrays, PyPlot

"""
return the initial condition evaluated on the given `domain`
"""
function initialCondition(domain::AbstractArray)
    return exp.(-5*(domain .- pi).^2)
end

"""
return the second order accurate central difference operator for the second
derivative with periodic boundary conditions. Matrix size is
`(number_of_nodes,number_of_nodes)`.
"""
function second_derivative_operator(dx::AbstractFloat, number_of_nodes::Int64)
    A = 1.0/dx^2*spdiagm(-1 => ones(number_of_nodes-1),
        0 => -2*ones(number_of_nodes), 1 => ones(number_of_nodes-1))
    A[1,number_of_nodes] = 1/dx^2
    A[number_of_nodes,1] = 1/dx^2
    return A
end

"""
return the right-hand-side of the reaction diffusion equation.
"""
function reactionDiffusionRHS(u::AbstractArray,
    diffusion_operator::SparseMatrixCSC)

    return diffusion_operator*u + u.^2
end

"""
perform one step of Forward Euler time integration for the given reaction
diffusion equation.
"""
function stepForwardEuler(u::AbstractArray, diffusion_operator::SparseMatrixCSC,
    dt::AbstractFloat)

    return u + dt*reactionDiffusionRHS(u, diffusion_operator)
end

"""
Use forward euler perform numerical integration up to the given
`stop_time` using the given time and space step sizes.
"""
function runForwardEuler(u0::AbstractArray, dt::AbstractFloat,
    stop_time::AbstractFloat, dx::AbstractFloat, number_of_nodes::Int64)

    diffusion_operator = second_derivative_operator(dx, number_of_nodes)
    current_time = 0.0
    u = copy(u0)
    while current_time <= stop_time
        current_time += dt
        u = stepForwardEuler(u, diffusion_operator, dt)
    end
    return u
end

"""
plot `u` against `domain`.
"""
function plotField(domain::AbstractArray, u::AbstractArray, u0::AbstractArray;
    ylabel = "", title = "")

    fig, ax = PyPlot.subplots()
    ax.plot(domain, u)
    ax.plot(domain, u0, "--")
    ax.set_xlabel(L"x")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid()
    return fig
end

"""
plot `num_iterations` treating data as a value per time step.
"""
function plotIterations(num_iterations::AbstractArray; ylabel = "", title = "")
    fig, ax = PyPlot.subplots()
    ax.plot(num_iterations, "-o")
    ax.set_xlabel("Number of iterations")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid()
    return fig
end

"""
return the Lagrange multiplier to construct the broyden matrix of the next step.
"""
function broydenLagrangeMultiplier(Bprev::Matrix, y::AbstractArray, s::AbstractArray)
    return (Bprev*y - s)/dot(y,y)
end

"""
return the Broyden matrix for the next step from the previous step.
"""
function updateBroydenMatrix(Bprev::Matrix, y::AbstractArray, s::AbstractArray)
    lambda = broydenLagrangeMultiplier(Bprev, y, s)
    return Bprev - lambda*y'
end

"""
return the non-linear function corresponding to the trapezoidal method that
we want to set to zero.
"""
function trapezoidNonlinearFunction(unext::AbstractArray, uprev::AbstractArray,
    diffusion_operator::SparseMatrixCSC, dt::AbstractFloat)

    Fprev = reactionDiffusionRHS(uprev, diffusion_operator)
    Fnext = reactionDiffusionRHS(unext, diffusion_operator)
    return unext - uprev - dt/2*(Fprev + Fnext)
end

"""
perform Broyden iterations until the nonlinear function satisfies the given
tolerance. Return the vector for the next time step.
"""
function iterateBroyden(u_previous_timestep::AbstractArray,
    diffusion_operator::SparseMatrixCSC, broydenMatrix::Matrix,
    dt::AbstractFloat; tol = 1e-8, maxiter = 100)

    uprev = copy(u_previous_timestep)
    unext = copy(uprev)
    count = 0
    Fprev = trapezoidNonlinearFunction(uprev, u_previous_timestep,
        diffusion_operator, dt)
    Fnext = copy(Fprev)

    while norm(Fnext) > tol && count < maxiter
        unext = uprev - broydenMatrix*Fprev
        Fnext = trapezoidNonlinearFunction(unext, u_previous_timestep,
            diffusion_operator, dt)
        s = unext - uprev
        y = Fnext - Fprev
        broydenMatrix = updateBroydenMatrix(broydenMatrix, y, s)
        Fprev = Fnext
        uprev = unext
        count += 1
    end
    if count == maxiter
        error("Broyden iterations failed to converge in $maxiter iterations")
    end
    return unext, broydenMatrix, count
end

function runTrapezoidal(u0::AbstractArray, dt::AbstractFloat,
    stop_time::AbstractFloat, dx::AbstractFloat, number_of_nodes::Int64;
    alpha = 0.05)

    diffusion_operator = second_derivative_operator(dx, number_of_nodes)
    current_time = 0.0
    u = copy(u0)
    broydenMatrix = alpha*Matrix(1.0*I, number_of_nodes, number_of_nodes)
    num_broyden_iterations = Int[]
    while current_time <= stop_time
        current_time += dt
        u, broydenMatrix, count = iterateBroyden(u, diffusion_operator, broydenMatrix, dt)
        push!(num_broyden_iterations, count)
    end
    return u, num_broyden_iterations
end


number_of_nodes = 101
stop_time = 6.0
dt = 0.1
dx = 2pi/(number_of_nodes-1)
domain = range(0.0, stop = 2pi, length = number_of_nodes)
u0 = initialCondition(domain)
diffusion_operator = second_derivative_operator(dx, number_of_nodes)

u, num_broyden_iterations = runTrapezoidal(u0, dt, stop_time, dx, number_of_nodes)

fig = plotField(domain, u, u0)
fig = plotIterations(num_broyden_iterations)

# alpha = 0.05
# B0 = alpha*Matrix(1.0I, number_of_nodes, number_of_nodes)
# u1, B1, count1 = iterateBroyden(u0, diffusion_operator, B0, 0.1)
# u2, B2, count2 = iterateBroyden(u1, diffusion_operator, B0, 0.1)
# u2, B2, count2_2 = iterateBroyden(u1, diffusion_operator, B1, 0.1)

# u = runSteps(stepForwardEuler, u0, dt, stop_time, dx, number_of_nodes)
# fig = plotField(domain, u, u0)
