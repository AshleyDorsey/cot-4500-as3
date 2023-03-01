# import libraries
import numpy as np

## Question 1 - Euler Method with the following details: function - t - y^2; range - 0 < t < 2; iterations - 10;
##              initial point - f(0) = 1

def function_given(t, y):
    return (t - y**2)

def eulers_method(t, y, iterations, x):
    h = x / iterations

    while t < x:
        y = y + h * function_given(t, y)
        t = t + h
    
    print(y, "\n")

## Question 2 - Runge-Kutta with the following details: function - t - y^2; range - 0 < t < 2; iterations - 10;
##              initial point - f(0) = 1

def func(t, y):
    return (t - y**2)

def runge_kutta(t, y, iterations, x):
    h = x / iterations
    
    while t < x:
        k_1 = h * func(t, y)
        k_2 = h * func((t + (h / 2)), (y + (k_1 / 2)))
        k_3 = h * func((t + (h / 2)), (y + (k_2 / 2)))
        k_4 = h * func((t + h), (y + k_3))

        y = y + (1 / 6) * (k_1 + (2 * k_2) + (2 * k_3) + k_4)
        t = t + h

    print(y, "\n")

## Question 3 - Use Gaussian elimination and backward substitution solve the linear system of equations written in
##              augmented matrix format

def gaussian_elimination(gaussian_matrix):
    n = 3
    singular_flag = forward_elimination(gaussian_matrix)

    backward_substitution(gaussian_matrix)

def swap_row(gaussian_matrix, i, j):
    n = 3
    for k in range(n + 1):
        temp = gaussian_matrix[i][k]
        gaussian_matrix[i][k] = gaussian_matrix[j][k]
        gaussian_matrix[j][k] = temp

def forward_elimination(gaussian_matrix):
    n = 3
    for k in range(n):
        i_max = k
        v_max = gaussian_matrix[i_max][k]

        for i in range(k + 1, n):
            if abs(gaussian_matrix[i][k]) > v_max:
                v_max = gaussian_matrix[i][k]
                i_max = i
        
        if not gaussian_matrix[k][i_max]:
            return k
        
        if i_max != k:
            swap_row(gaussian_matrix, k, i_max)

        for i in range(k + 1, n):
            f = gaussian_matrix[i][k] / gaussian_matrix[k][k]
            for j in range(k + 1, n + 1):
                gaussian_matrix[i][j] -= gaussian_matrix[k][j] * f
            
            gaussian_matrix[i][k] = 0
    return -1
        
def backward_substitution(gaussian_matrix):
    n = 3
    x = [None for _ in range(n)]

    for i in range(n - 1, -1, -1):
        x[i] = gaussian_matrix[i][n]
        for j in range(i + 1, n):
            x[i] -= gaussian_matrix[i][j] * x[j]

        x[i] = x[i] / gaussian_matrix[i][i]

    for i in range(n):
        print(x[i], "\n")


## Question 4 - Implement LU Factorization for the following matrix and do the following:
##              a) print out the matrix determinant
##              b) print out the L matrix
##              c) print out the U matrix

def lu_factorization(lu_matrix, n):
    l = [[0 for x in range(n)] for y in range(n)]
    u = [[0 for x in range(n)] for y in range(n)]

    for i in range(n):
        for k in range(i, n):
            sum = 0
            for j in range(i):
                sum += l[i][j] * u[j][k]
            
            u[i][k] = lu_matrix[i][k] - sum
        
        for k in range(i, n):
            if i == k:
                l[i][i] = 1
            else:
                sum = 0
                for j in range(i):
                    sum += l[k][j] * u[j][i]

                l[k][i] = (lu_matrix[k][i] - sum) / u[i][i]

## Question 5 - Determine if the following matrix is diagonally dominate (true/false)

def diagonally_dominant(dd_matrix, n):
    for i in range(0, n):
        sum = 0
        for j in range(0, n):
            sum = sum + abs(dd_matrix[i][j])
        
        sum = sum - abs(dd_matrix[i][i])
    
    if abs(dd_matrix[i][i]) < sum:
        print("False\n")
    else:
        print("True\n")

## Question 6 - Determine if the matrix is a positive definite (true/false)

def positive_definite(pd_matrix):
    if np.all(np.linalg.eigvals(pd_matrix) > 0):
        print("True\n")
    else:
        print("False\n")

# main function
if __name__ == "__main__":
    
    ## 1 - Euler's Method
    t_0 = 0
    y_0 = 1
    iterations = 10
    x = 2
    eulers_method(t_0, y_0, iterations, x) # should be 1.244638

    ## 2 - Runge-Kutta 
    t_0 = 0
    y_0 = 1
    iterations = 10
    x = 2
    runge_kutta(t_0, y_0, iterations, x) # should be 1.251316

    ## 3 - Gaussian elimination and backwards substitution - should be [2 -1 1]
    gaussian_matrix = [[2, -1, 1, 6], [1, 3, 1, 0], [-1, 5, 4, -3]]
    gaussian_elimination(gaussian_matrix)

    ## 4 - LU factorization - should be 38.9999999, matrix, matrix
    lu_matrix = [[1, 1, 0, 3], [2, 1, -1, 1], [3, -1, -1, 2], [-1, 2, 3, -1]]
    n = 4
    lu_factorization(lu_matrix, n)

    ## 5 - Diagonally dominate - should be false
    n = 5
    dd_matrix =[[9, 0, 5, 2, 1], [3, 9, 1, 2, 1], [0, 1, 7, 2, 3], [4, 2, 3, 12, 2], [3, 2, 4, 0, 8]]
    diagonally_dominant(dd_matrix, n)

    ## 6 - Positive definite - should be true
    pd_matrix = np.matrix([[2, 2, 1], [2, 3, 0], [1, 0, 2]])
    positive_definite(pd_matrix)