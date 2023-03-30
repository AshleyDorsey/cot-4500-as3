# import libraries
import numpy as np

## Question 1 - Euler Method with the following details: function - t - y^2; range - 0 < t < 2; iterations - 10;
##              initial point - f(0) = 1

# create a function to be the function the question gives
def function_given(t, y):
    # return the function
    return (t - y**2)

# create a function to calculate the approximate value based on Euler's Method
def eulers_method(t, y, iterations, x):
    # find the step by subtracting the final range from the starting point and dividing by the number of iterations
    h = (x - t) / iterations

    # use a for loop to find the final value of the approximate value
    # in this case, we need to use a random variable to make the loop work, but don't use y or t because it will lead
    #   to the wrong value
    # use the range of iterations to be what you loop through
    for unused_variable in range(iterations):
        # set y equal to y plus the step times by what the function gives with the current y and t
        y = y + (h * function_given(t, y))

        # set t equal to t plus the step to continue getting a new t
        t = t + h
    
    # print the final y, which is the approximate value
    print(y, "\n")

## Question 2 - Runge-Kutta with the following details: function - t - y^2; range - 0 < t < 2; iterations - 10;
##              initial point - f(0) = 1

# create a function that is the function we're given
def func(t, y):
    # return the function
    return (t - y**2)

# create a function to calculate the approximate value using the Runge-Kutta method
def runge_kutta(t, y, iterations, x):
    # find the step by subtracting the range from the initial point and dividing by the iterations
    h = (x - t) / iterations
    
    # use a for loop to find the final value of y
    # use another random variable to make the for loop work, but it won't be y or t or else the final value won't
    #   be correct; we mainly just need it to make the for loop work
    # use the range of the iterations to loop through
    for another_unused_variable in range(iterations):
        # find all of the k1 - k4 values (this is in the notes powerpoint)
        k_1 = h * func(t, y)
        k_2 = h * func((t + (h / 2)), (y + (k_1 / 2)))
        k_3 = h * func((t + (h / 2)), (y + (k_2 / 2)))
        k_4 = h * func((t + h), (y + k_3))

        # set y equal to the following (this was also found in the powerpoint)
        y = y + (1 / 6) * (k_1 + (2 * k_2) + (2 * k_3) + k_4)

        # set t equal to t plus the step to continue getting a new t
        t = t + h

    # print the final y
    print(y, "\n")

## Question 3 - Use Gaussian elimination and backward substitution solve the linear system of equations written in
##              augmented matrix format

# create a function to find the solution to the matrix using Gaussian elimination and backward substitution
def gaussian_elimination(gaussian_matrix):
    # set the size equal to the matrix initial shape
    size = gaussian_matrix.shape[0]

    # create a for loop
    for i in range(size):
        # set the pivot to i (this means it will go up with each loop)
        pivot = i
        # while matrix at the point pivot, i equals 0, increase the pivot by 1
        while gaussian_matrix[pivot, i] == 0:
            # pivot plus equal to 1
            pivot += 1

        # set the matrix[i, pivot] equal to the matrix[pivot, i]
        gaussian_matrix[[i, pivot]] = gaussian_matrix[[pivot, i]]

        # create another foor loop
        for j in range(i + 1, size):
            # set the factor equal to the matrix[j, i] divided by the matrix[i, i]
            factor = gaussian_matrix[j, i] / gaussian_matrix[i, i]
            # set the matrix[j, all of i] equal to the matrix[j, all of i] minus the factor times the 
            #   matrix[i, all of i]
            gaussian_matrix[j, i:] = gaussian_matrix[j, i:] - factor * gaussian_matrix[i, i:]

    # set a variable equal to the size of the size (creating an array)
    inputs = np.zeros(size)

    # create another for loop to find the final values
    for i in range(size - 1, -1, -1):
        # set the variable equal to the matrix[i, -1] minus the matrix[i, all of i starting at the last value]
        #   and the variable[all of i] divided by the matrix[i, i]
        inputs[i] = (gaussian_matrix[i, -1] - np.dot(gaussian_matrix[i, i: -1], inputs[i:])) / gaussian_matrix[i, i]
    
    # set another variable equal to an array of the first three inputs (all of which need to be int)
    final_answer = np.array([int(inputs[0]), int(inputs[1]), int(inputs[2])])
    # print the final answer
    print(final_answer, "\n")

## Question 4 - Implement LU Factorization for the following matrix and do the following:
##              a) print out the matrix determinant
##              b) print out the L matrix
##              c) print out the U matrix

# create a function to do the LU factorization
def lu_factorization(lu_matrix):
    # set the size equal to the matrix initial shape
    size = lu_matrix.shape[0]

    # set the l equal to the size, but where the left over values will be 0s
    l_factor = np.eye(size)
    # set the u equal to the size, but where the left over values will be 0s
    u_factor = np.zeros_like(lu_matrix)

    # create a for loop
    for i in range(size):
        # create another step in the for loop to find the u
        for j in range(i, size):
            # set the u[i, j] equal to the matrix[i, j] minus the l[i, all of i] and u[all of i, j]
            u_factor[i, j] = (lu_matrix[i, j] - np.dot(l_factor[i, :i], u_factor[:i, j]))

        # create another step in the for loop to find the l
        for j in range(i + 1, size):
            # set the l[j, i] equal to the matrix[j, i] minus the l[j, all of i] and u[all of i, i] all
            #   divided by u[i, i]
            l_factor[j, i] = (lu_matrix[j, i] - np.dot(l_factor[j, :i], u_factor[:i, i])) / u_factor[i, i]
    
    # set the determinant equal to the linear algebra determinant function of the matrix
    determinant = np.linalg.det(lu_matrix)

    # print the determinant
    print(determinant, "\n")
    # print the l matrix
    print(l_factor, "\n")
    # print the u matrix
    print(u_factor, "\n")

## Question 5 - Determine if the following matrix is diagonally dominate (true/false)

# create a function to figure out if the matrix is diagonally dominant
def diagonally_dominant(dd_matrix, n):

    # create a for loop
    for i in range(0, n):
        # set the total equal to 0 initially
        total = 0
        # create another step in the for loo[]
        for j in range(0, n):
            # set the total equal to the current total plus the absolute value of the matrix at [i][j]
            total = total + abs(dd_matrix[i][j])
        
        # set the total equal to the total minus the absolute value of the matrix at [i][i]      
        total = total - abs(dd_matrix[i][i])
    
    # if the absolute value of the matrix at [i][i] is less than the total
    if abs(dd_matrix[i][i]) < total:
        # print false
        print("False\n")
    # otherwise, if the absolute value is greater than or equal to the total
    else:
        # print true
        print("True\n")

## Question 6 - Determine if the matrix is a positive definite (true/false)

# create a function to figure out if the matrix is positive definite
def positive_definite(pd_matrix):
    # set the eigenvalues variable equal to the linear algebra eigvals function of the matrix
    eigenvalues = np.linalg.eigvals(pd_matrix)

    # if all of the eigenvalues are greater than 0
    if np.all(eigenvalues > 0):
        # print true
        print("True\n")
    # otherwise if not all of the eigenvalues are greater than 0
    else:
        # print false
        print("False\n")

# main function
if __name__ == "__main__":
    
    ## 1 - Euler's Method
    # set the initial x-value to be 0
    t_0 = 0
    # set the initial y-value to be 1
    y_0 = 1
    # set the number of iterations to be 10
    iterations = 10
    # set the range to be the final range minus the initial range (2)
    x = 2
    # call function
    eulers_method(t_0, y_0, iterations, x)

    ## 2 - Runge-Kutta 
    # set the initial x-value to be 0
    t_0 = 0
    # set the initial y-valye to be 1
    y_0 = 1
    # set the number of iterations to be 10
    iterations = 10
    # set the range to be the final range minus the initial range (2)
    x = 2
    # call function
    runge_kutta(t_0, y_0, iterations, x) 

    ## 3 - Gaussian elimination and backwards substitution 
    # create the matrix
    gaussian_matrix = np.array([[2, -1, 1, 6], [1, 3, 1, 0], [-1, 5, 4, -3]])
    # call the function
    gaussian_elimination(gaussian_matrix)

    ## 4 - LU factorization 
    # create the matrix (make sure it has the type double)
    lu_matrix = np.array([[1, 1, 0, 3], [2, 1, -1, 1], [3, -1, -1, 2], [-1, 2, 3, -1]], dtype = np.double)
    # call the function
    lu_factorization(lu_matrix)

    ## 5 - Diagonally dominate 
    # set n (the size) equal to the amount of rows and columns (5)
    n = 5
    # create the matrix
    dd_matrix = np.array([[9, 0, 5, 2, 1], [3, 9, 1, 2, 1], [0, 1, 7, 2, 3], [4, 2, 3, 12, 2], [3, 2, 4, 0, 8]])
    # call the function
    diagonally_dominant(dd_matrix, n)

    ## 6 - Positive definite
    # create the matrix
    pd_matrix = np.matrix([[2, 2, 1], [2, 3, 0], [1, 0, 2]])
    # call the function
    positive_definite(pd_matrix)