# import libraries
import numpy as np

## Question 1 - Euler Method with the following details: function - t - y^2; range - 0 < t < 2; iterations - 10;
##              initial point - f(0) = 1

def function_given(t, y):
    return (t - y**2)

def eulers_method(t, y, iterations, x):
    h = (x - t) / iterations

    for unused_variable in range(iterations):
        y = y + (h * function_given(t, y))
        t = t + h
    
    print(y, "\n")

## Question 2 - Runge-Kutta with the following details: function - t - y^2; range - 0 < t < 2; iterations - 10;
##              initial point - f(0) = 1

def func(t, y):
    return (t - y**2)

def runge_kutta(t, y, iterations, x):
    h = (x - t) / iterations
    
    for another_unused_variable in range(iterations):
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
    size = gaussian_matrix.shape[0]

    for i in range(size):
        pivot = i
        while gaussian_matrix[pivot, i] == 0:
            pivot += 1
    
        gaussian_matrix[[i, pivot]] = gaussian_matrix[[pivot, i]]

        for j in range(i + 1, size):
            factor = gaussian_matrix[j, i] / gaussian_matrix[i, i]
            gaussian_matrix[j, i:] = gaussian_matrix[j, i:] - factor * gaussian_matrix[i, i:]

    inputs = np.zeros(size)

    for i in range(size - 1, -1, -1):
        inputs[i] = (gaussian_matrix[i, -1] - np.dot(gaussian_matrix[i, i: -1], inputs[i:])) / gaussian_matrix[i, i]
    
    final_answer = np.array([int(inputs[0]), int(inputs[1]), int(inputs[2])])
    print(final_answer, "\n")

## Question 4 - Implement LU Factorization for the following matrix and do the following:
##              a) print out the matrix determinant
##              b) print out the L matrix
##              c) print out the U matrix

def lu_factorization(lu_matrix):
    size = lu_matrix.shape[0]

    l_factor = np.eye(size)
    u_factor = np.zeros_like(lu_matrix)

    for i in range(size):
        for j in range(i, size):
            u_factor[i, j] = (lu_matrix[i, j] - np.dot(l_factor[i, :i], u_factor[:i, j]))
    
        for j in range(i + 1, size):
            l_factor[j, i] = (lu_matrix[j, i] - np.dot(l_factor[j, :i], u_factor[:i, i])) / u_factor[i, i]
    
    determinant = np.linalg.det(lu_matrix)

    print(determinant, "\n")
    print(l_factor, "\n")
    print(u_factor, "\n")

## Question 5 - Determine if the following matrix is diagonally dominate (true/false)

def diagonally_dominant(dd_matrix, n):

    for i in range(0, n):
        total = 0
        for j in range(0, n):
            total = total + abs(dd_matrix[i][j])
        
        total = total - abs(dd_matrix[i][i])
    
    if abs(dd_matrix[i][i]) < total:
        print("False\n")
    else:
        print("True\n")

## Question 6 - Determine if the matrix is a positive definite (true/false)

def positive_definite(pd_matrix):
    eigenvalues = np.linalg.eigvals(pd_matrix)

    if np.all(eigenvalues > 0):
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
    eulers_method(t_0, y_0, iterations, x)

    ## 2 - Runge-Kutta 
    t_0 = 0
    y_0 = 1
    iterations = 10
    x = 2
    runge_kutta(t_0, y_0, iterations, x) 

    ## 3 - Gaussian elimination and backwards substitution 
    gaussian_matrix = np.array([[2, -1, 1, 6], [1, 3, 1, 0], [-1, 5, 4, -3]])
    gaussian_elimination(gaussian_matrix)

    ## 4 - LU factorization 
    lu_matrix = np.array([[1, 1, 0, 3], [2, 1, -1, 1], [3, -1, -1, 2], [-1, 2, 3, -1]], dtype = np.double)
    lu_factorization(lu_matrix)

    ## 5 - Diagonally dominate 
    n = 5
    dd_matrix = np.array([[9, 0, 5, 2, 1], [3, 9, 1, 2, 1], [0, 1, 7, 2, 3], [4, 2, 3, 12, 2], [3, 2, 4, 0, 8]])
    diagonally_dominant(dd_matrix, n)

    ## 6 - Positive definite
    pd_matrix = np.matrix([[2, 2, 1], [2, 3, 0], [1, 0, 2]])
    positive_definite(pd_matrix)