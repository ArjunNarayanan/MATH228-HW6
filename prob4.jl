using LinearAlgebra
using SparseArrays
using PyPlot

function givensCoefficients(sub_diagonal_entry::Real, diagonal_entry::Real)
    theta = atan(sub_diagonal_entry, diagonal_entry)
    cosine = cos(theta)
    sine = sin(theta)
    return cosine, sine
end

# function givensCoefficients(sub_diagonal_entry::Complex, diagonal_entry::Complex)
#     s0 = conj(sub_diagonal_entry)
#     c0 = conj(diagonal_entry)
#     magnitude = sqrt(norm(c0)^2 + norm(s0)^2)
#     c = c0/magnitude
#     s = s0/magnitude
#     return c, s
# end

function givensCoefficients(sub_diagonal_entry::Complex, diagonal_entry::Complex)
    theta = atan(sub_diagonal_entry/diagonal_entry)
    cosine = cos(theta)
    sine = sin(theta)
    return cosine, sine
end

function updateRows(matrix::Matrix, cosine::Real, sine::Real, row)
    num_rows, num_cols = size(matrix)
    for col in 1:num_cols
        upper_row_val = matrix[row,col]
        lower_row_val = matrix[row+1,col]
        matrix[row,col] = upper_row_val*cosine + lower_row_val*sine
        matrix[row+1,col] = -upper_row_val*sine + lower_row_val*cosine
    end
end

function updateRows(matrix::Matrix, cosine::Complex, sine::Complex, row)
    num_rows, num_cols = size(matrix)
    for col in 1:num_cols
        upper_row_val = matrix[row,col]
        lower_row_val = matrix[row+1,col]
        matrix[row,col] = upper_row_val*cosine + lower_row_val*sine
        matrix[row+1,col] = -upper_row_val*conj(sine) + lower_row_val*conj(cosine)
    end
end

function updateRows(rhs::Vector, cosine::Real, sine::Real, row)
    upper_row_val = rhs[row]
    lower_row_val = rhs[row+1]

    rhs[row] = upper_row_val*cosine + lower_row_val*sine
    rhs[row+1] = -upper_row_val*sine + lower_row_val*cosine
end

function updateRows(rhs::Vector, cosine::Complex, sine::Complex, row)
    upper_row_val = rhs[row]
    lower_row_val = rhs[row+1]

    rhs[row] = upper_row_val*cosine + lower_row_val*sine
    rhs[row+1] = -upper_row_val*conj(sine) + lower_row_val*conj(cosine)
end

function solveRow(A::Matrix, solution::Vector{Float64}, rhs::Vector{Float64}, row, num_cols)
        solution[row] = (rhs[row] -
                     dot(A[row,row+1:num_cols],solution[row+1:num_cols]))/A[row,row]
end

function solveRow(A::Matrix, solution::Vector{ComplexF64}, rhs::Vector{ComplexF64}, row, num_cols)
        solution[row] = (rhs[row] -
                     sum(A[row,row+1:num_cols] .* solution[row+1:num_cols]))/A[row,row]
end

function backSolve(A::Matrix, rhs::Array{Float64, 1})
    num_rows, num_cols = size(A)
    solution = zeros(num_cols)
    solution[num_cols] = rhs[num_cols]/A[num_cols,num_cols]
    for i in 1:num_cols - 1
        row = num_cols - i
        solveRow(A,solution,rhs,row,num_cols)
    end
    return solution
end

function backSolve(A::Matrix, rhs::Array{ComplexF64, 1})
    num_rows, num_cols = size(A)
    solution = zeros(ComplexF64, num_cols)
    solution[num_cols] = rhs[num_cols]/A[num_cols,num_cols]
    for i in 1:num_cols - 1
        row = num_cols - i
        solveRow(A,solution,rhs,row,num_cols)
    end
    return solution
end

function givensQR!(matrix::Matrix, rhs::Vector)
    num_rows, num_cols = size(matrix)
    @assert (num_rows == num_cols + 1) "Matrix dimensionality must by (m+1,m), got ($num_rows,$num_cols)"
    for col in 1:num_cols
        diagonal_entry = matrix[col,col]
        sub_diagonal_entry = matrix[col+1,col]
        cosine, sine = givensCoefficients(sub_diagonal_entry, diagonal_entry)
        updateRows(matrix, cosine, sine, col)
        updateRows(rhs, cosine, sine, col)
    end
    residual = abs(rhs[num_rows])
    return residual
end

function testMatrix(number_of_rows)
    number_of_columns = number_of_rows - 1
    M = zeros(number_of_rows, number_of_columns)
    M[1,:] = rand(number_of_columns)
    for row in 2:number_of_rows
        M[row, row-1:end] = rand(number_of_columns - row + 2)
    end
    return M
end

function testMatrixComplex(number_of_rows)
    number_of_columns = number_of_rows - 1
    M = zeros(ComplexF64, number_of_rows, number_of_columns)
    M[1,:] = rand(ComplexF64, number_of_columns)
    for row in 2:number_of_rows
        M[row, row-1:end] = rand(ComplexF64, number_of_columns - row + 2)
    end
    return M
end

function solveGivensQR(A,rhs)
    givensQR!(A,rhs)
    return backSolve(A,rhs)
end

function arnoldi(A::SparseMatrixCSC, v::Vector{T}, m) where T
    rows, cols = size(A)
    orthonormal_vectors = zeros(T, rows, m+1)
    hessenberg_matrix = zeros(T, m+1,m)
    orthonormal_vectors[:,1] = v/norm(v)
    for j in 1:m
        vj = orthonormal_vectors[:,j]
        wj = A*vj
        for i in 1:j
            vi = orthonormal_vectors[:,i]
            hij = dot(vi,wj)
            wj = wj - hij*vi
            hessenberg_matrix[i,j] = hij
        end
        h = norm(wj)
        if isapprox(h,0.0, atol = 1e-15)
            m = j
            break
        end
        hessenberg_matrix[j+1,j] = h
        orthonormal_vectors[:,j+1] = wj/h
    end
    return orthonormal_vectors, hessenberg_matrix, m
end

function solveLeastSquaresDirect(A::Matrix, rhs::Vector)
    return (A'*A)\(A'*rhs)
end

function stepGMRES(A::SparseMatrixCSC, guess::Vector, residual::Vector, subspace_dim)

    beta = norm(residual)
    subspace_basis, hessenberg_matrix, subspace_dim = arnoldi(A, residual, subspace_dim)
    e = zeros(Complex{Float64}, subspace_dim+1)
    e[1] = beta
    H = hessenberg_matrix[1:subspace_dim+1,1:subspace_dim]
    coefficients = solveLeastSquaresDirect(H,e)
    # coefficients = solveGivensQR(H,e)
    V = subspace_basis[:,1:subspace_dim]
    return guess + V*coefficients
end

function runGMRES(A::SparseMatrixCSC, guess::Vector, rhs::Vector,
    niter, subspace_dim)

    solution = copy(guess)
    residual = rhs - A*solution
    for iterations in 1:niter
        solution = stepGMRES(A, solution, residual, subspace_dim)
        residual = rhs - A*solution
    end
    return solution
end

function runGMRES(A::SparseMatrixCSC, guess::Vector, rhs::Vector;
    tol = 1e-12, maxiter = 100, subspace_dim = 20)

    solution = copy(guess)
    residual = rhs - A*solution
    count = 0
    while norm(residual) > tol && count < maxiter
        count = count + 1
        solution = stepGMRES(A, solution, residual, subspace_dim)
        residual = rhs - A*solution
    end
    if count == maxiter
        @warn "GMRES failed to converge in $maxiter iterations"
    end
    println("\t #GMRES iterations = ", count)
    return solution
end

function analyticalSolution(x, t)
    return 2.0*a/β * sech(sqrt(a)*(x-x0-c*t)) * exp(im*(c/2.0*(x-x0)-(c^2/4.0-a)*t))
end

function plot_field(x,u)
    fig, ax = PyPlot.subplots()
    ax.plot(x,u)
    ax.grid()
    return fig
end

function plot_field(x,u,u0)
    fig, ax = PyPlot.subplots()
    ax.plot(x,u, linewidth = 2.0)
    ax.plot(x,u0, "-")
    ax.grid()
    return fig
end

function periodicLaplacian(hs, Ns)
    laplacian = 1.0/hs^2 * spdiagm(-1=> ones(Ns-1),0=> -2.0*ones(Ns),1=> ones(Ns-1))
    return laplacian
end

function nonlinearSchrodingerRHS(u::Vector)
    return -im*β*((norm.(u)).^2).*u
end

function schrodingerRHS(u::Vector, laplacian::SparseMatrixCSC)
    return im*laplacian*u + nonlinearSchrodingerRHS(u)
end

function IMEX_rhs_AB2(u1::Vector, u0::Vector, laplacian::SparseMatrixCSC, dt)
    return u1 + dt/2*im*laplacian*u1 + dt/2*(3*nonlinearSchrodingerRHS(u1) -
                                       nonlinearSchrodingerRHS(u0))
end

function IMEX_rhs_Euler(u0::Vector, laplacian::SparseMatrixCSC, dt)
    return u0 + dt/2*im*laplacian*u0 + dt*nonlinearSchrodingerRHS(u0)
end

function IMEX_matrix(laplacian::SparseMatrixCSC, dt)
    return I - dt/2*im*laplacian
end

function solveDirect(A, rhs)
    return A\rhs
end

function stepIMEX(imex_matrix::SparseMatrixCSC,
    laplacian::SparseMatrixCSC, u1::Vector, u0::Vector, dt)

    imex_rhs = IMEX_rhs_AB2(u1, u0, laplacian, dt)
    return solveDirect(imex_matrix, imex_rhs)
end

function runStepsIMEX(u0::Vector, laplacian::SparseMatrixCSC,
    dt::AbstractFloat, stop_time::AbstractFloat)

    ndofs = length(u0)
    imex_matrix = IMEX_matrix(laplacian, dt)
    imex_rhs = IMEX_rhs_Euler(u0, laplacian, dt)
    # u1 = solveDirect(imex_matrix, imex_rhs)
    u1 = runGMRES(imex_matrix, u0, imex_rhs)
    u2 = copy(u1)
    current_time = dt
    while current_time < stop_time
        println("Time = ", current_time)
        current_time += dt
        imex_rhs = IMEX_rhs_AB2(u1, u0, laplacian, dt)
        # u2 = solveDirect(imex_matrix, imex_rhs)
        u2 = runGMRES(imex_matrix, u1, imex_rhs)
        u0 = u1
        u1 = u2
    end
    return u2
end

function rk4Stages(solution::Vector, laplacian::SparseMatrixCSC,
    step_size::AbstractFloat)

    k1 = schrodingerRHS(solution, laplacian)
    k2 = schrodingerRHS(solution+step_size/2*k1, laplacian)
    k3 = schrodingerRHS(solution+step_size/2*k2, laplacian)
    k4 = schrodingerRHS(solution+step_size*k3, laplacian)
    return k1, k2, k3, k4
end

function stepRK4(solution::Vector, laplacian::SparseMatrixCSC,
    step_size::AbstractFloat)

    k1, k2, k3, k4 = rk4Stages(solution, laplacian, step_size)
    return solution + step_size/6*(k1+2k2+2k3+k4)
end

function runStepsRK4(solution0::Vector, laplacian::SparseMatrixCSC,
    step_size::AbstractFloat, stop_time::AbstractFloat)

    ndofs = length(solution0)
    solution = copy(solution0)
    current_time = 0.0
    while current_time < stop_time
        solution = stepRK4(solution, laplacian, step_size)
        current_time += step_size
    end
    return solution
end

function computeError(u_numeric, u_analytic)
    return maximum(abs.(u_numeric - u_analytic))
end

const L = 20.0
const stop_time = 1.0
const β = -8.0
const a = β^2/16.0
const c = 0.5
const x0 = -3.0

number_of_nodes = 1500
dx = L/(number_of_nodes - 1)
dt_rk4 = 5e-5
dt_imex = 5e-5


M = testMatrix(10)
rhs = rand(10)
givensQR!(M,rhs)


# domain = range(-L/2, stop = L/2, length = number_of_nodes)
# initial_condition = analyticalSolution.(domain,0.0)
# u0 = initial_condition
# analytical_solution = analyticalSolution.(domain,stop_time)
# #
# laplacian = periodicLaplacian(dx, number_of_nodes)


# imex_matrix = IMEX_matrix(laplacian, dt_imex)
# imex_rhs = IMEX_rhs_Euler(u0, laplacian, dt_imex)
#
# u1_gmres = runGMRES(imex_matrix, u0, imex_rhs)
#
# imex_rhs = IMEX_rhs_AB2(u1_gmres, u0, laplacian, dt_imex)
#
# u2_direct = solveDirect(imex_matrix, imex_rhs)
# u2_gmres = runGMRES(imex_matrix, u1_gmres, imex_rhs)
#
# println("Error = ", norm(u2_gmres - u2_direct))

# imex_solution = runStepsIMEX(u0, laplacian, dt_imex, stop_time)
# rk4_solution = runStepsRK4(u0, laplacian, dt_rk4, stop_time)
# println("Max error IMEX = ", computeError(imex_solution, analytical_solution))
# println("Max error RK4  = ", computeError(rk4_solution, analytical_solution))
# fig = plot_field(domain,real(u))
