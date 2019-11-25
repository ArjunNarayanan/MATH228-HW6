using LinearAlgebra, SparseArrays, PyPlot

function ndgrid(v1::AbstractVector{T}, v2::AbstractVector{T}) where T
    m, n = length(v1), length(v2)
    v1 = reshape(v1, m, 1)
    v2 = reshape(v2, 1, n)
    (repeat(v1, 1, n), repeat(v2, m, 1))
end

function constructDomain(number_of_nodes::Int64)
    dx = 1.0/(number_of_nodes - 1)
    @assert isinteger((number_of_nodes-1)*0.5) "Require grid size such that 0.5/h is integer, got $dx"
    domain_x = 0.0:dx:1.0
    X, Y = ndgrid(domain_x, domain_x)
    return X, Y
end

function inCutOut(I::Int64, J::Int64, M::Int64)
    if I >= M && J >= M
        return true
    else
        return false
    end
end

function onBoundary(I::Int64, J::Int64, nx::Int64)
    if I == 1 || I == nx || J == 1 || J == nx
        return true
    else
        return false
    end
end

function mask(nx::Int64)
    mid = round(Int,(nx+1)/2)
    U = zeros(nx,nx)
    for j in 1:nx
        for i in 1:nx
            U[i,j] = inCutOut(i,j,mid) || onBoundary(i,j,nx) ? 0.0 : 1.0
        end
    end
    return U
end

function indexToDOF(I::Int64, J::Int64, nx::Int64)
    return (I-1)*nx + J
end

"""
return the poisson operator on a grid of size `(nx,nx)` with
step size `dx`. The operator incorporates boundary conditions and the
presence of the cutout.
"""
function poissonOperator(nx::Int64, dx::AbstractFloat)
    rows = Int64[]
    cols = Int64[]
    vals = Float64[]
    ndofs = nx*nx
    stencil = -[1.0, 1.0, -4.0, 1.0, 1.0]/dx^2
    mid = round(Int,(nx+1)/2)
    for I in 1:nx
        for J in 1:nx
            current_dof = indexToDOF(I, J, nx)
            if inCutOut(I, J, mid) || onBoundary(I, J, nx)
                append!(rows, current_dof)
                append!(cols, current_dof)
                append!(vals, 1.0)
            else
                append!(rows, repeat([current_dof], 5))
                right = indexToDOF(I, J+1, nx)
                left = indexToDOF(I, J-1, nx)
                top = indexToDOF(I+1, J, nx)
                down = indexToDOF(I-1, J, nx)
                append!(cols, [right, left, current_dof, top, down])
                append!(vals, stencil)
            end
        end
    end
    A = sparse(rows, cols, vals, ndofs, ndofs)
    return A
end

"""
construct the rght-hand-side vector for the given poisson equation.
"""
function poissonRHS(nx::Int64, dx::AbstractFloat)
    ndofs = nx*nx
    rhs = zeros(ndofs)
    mid = round(Int,(nx+1)/2)
    for I in 1:nx
        for J in 1:nx
            current_dof = indexToDOF(I, J, nx)
            if inCutOut(I, J, mid) || onBoundary(I, J, nx)
                rhs[current_dof] = 0.0
            else
                rhs[current_dof] = 1.0
            end
        end
    end
    rhs
end

"""
return `x` such that `A*x = rhs` using conjugate gradient iterations.
"""
function solveCG(A::SparseMatrixCSC, rhs::AbstractArray; tol = 1e-8)
    ndofs = length(rhs)
    solution = copy(rhs)
    residual = rhs - A*solution
    update_direction = copy(residual)
    count = 0
    while norm(residual) > tol && count <= ndofs
        alpha = update_direction'*residual/(update_direction'*A*update_direction)
        solution += alpha*update_direction
        residual = rhs - A*solution
        beta = update_direction'*A*residual/(update_direction'*A*update_direction)
        update_direction = residual + beta*update_direction
        count += 1
    end
    if count == ndofs
        error("CG failed to converge in ndofs iterations")
    end
    return solution, count
end

"""
return `x` such that `A*x = rhs` using conjugate gradient iterations.
"""
function solveEfficientCG(A::SparseMatrixCSC, rhs::AbstractArray; tol = 1e-8)
    ndofs = length(rhs)
    solution = copy(rhs)
    residual = rhs - A*solution
    update_direction = copy(residual)
    count = 0
    while norm(residual) > tol && count <= ndofs

        Ap0 = A*update_direction
        p0Ap0 = update_direction'*Ap0

        alpha = update_direction'*residual/(p0Ap0)
        solution += alpha*update_direction

        residual -= alpha*Ap0
        beta = residual'*Ap0/p0Ap0

        update_direction *= beta
        update_direction += residual

        count += 1
    end
    if count == ndofs
        error("CG failed to converge in $ndofs iterations")
    end
    return solution, count
end

number_of_nodes = 201
dx = 1.0/(number_of_nodes-1)
A = poissonOperator(number_of_nodes, dx)
rhs = poissonRHS(number_of_nodes, dx)
@time U = A\rhs
@time U_CG, countCG = solveCG(A, rhs)
@time U_eff_CG, count_effCG = solveEfficientCG(A, rhs)

err1 = norm(U - U_CG)
err2 = norm(U - U_eff_CG)
println("Errors: $err1, $err2")

# U = reshape(U, number_of_nodes, number_of_nodes)
# U_CG = reshape(U_CG, number_of_nodes, number_of_nodes)
# U_eff_CG = reshape(U_eff_CG, number_of_nodes, number_of_nodes)
# X, Y = constructDomain(number_of_nodes)
# fig, ax = PyPlot.subplots()
# ax.contourf(X,Y,U_eff_CG)
# fig
