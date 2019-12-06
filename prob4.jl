using LinearAlgebra
using SparseArrays

function givensTheta(sub_diagonal_entry, diagonal_entry)
    return atan(sub_diagonal_entry, diagonal_entry)
end

function updateRows(matrix::Matrix, theta, row)
    cosine = cos(theta)
    sine = sin(theta)
    num_rows, num_cols = size(matrix)
    for col in 1:num_cols
        upper_row_val = matrix[row,col]
        lower_row_val = matrix[row+1,col]
        matrix[row,col] = upper_row_val*cosine + lower_row_val*sine
        matrix[row+1,col] = -upper_row_val*sine + lower_row_val*cosine
    end
end

function updateRows(rhs::Vector, theta, row)
    cosine = cos(theta)
    sine = sin(theta)
    upper_row_val = rhs[row]
    lower_row_val = rhs[row+1]

    rhs[row] = upper_row_val*cosine + lower_row_val*sine
    rhs[row+1] = -upper_row_val*sine + lower_row_val*cosine
end

function givensQR(matrix::Matrix, rhs::Vector)
    num_rows, num_cols = size(matrix)
    @assert (num_rows == num_cols + 1) "Matrix dimensionality must by (m+1,m), got ($num_rows,$num_cols)"
    for col in 1:num_cols
        diagonal_entry = matrix[col,col]
        sub_diagonal_entry = matrix[col+1,col]
        theta = givensTheta(sub_diagonal_entry, diagonal_entry)
        updateRows(matrix, theta, col)
        updateRows(rhs, theta, col)
    end
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

ndofs = 10
A = testMatrix(ndofs)
rhs = rand(ndofs)
givensQR(A,rhs)
