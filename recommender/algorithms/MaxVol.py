import numpy as np
from numpy import linalg as LA

def find_solution(A : np.ndarray):
    """
    implementation of paper: How to find a good submatrix
    """
    rows, cols = A.shape

    submatrix = A[np.sort(np.random.choice(rows, cols, replace=False))]
    print(submatrix)

    deleted = 0
    for row_in_subA in range(0, cols):
        for row_in_A in range(0, rows - deleted):
            if np.array_equal(A[row_in_A], submatrix[row_in_subA]):
                A = np.delete(A, row_in_A, axis=0)
                deleted += 1
                break

    new_A = np.concatenate((submatrix, A))

    B = np.matmul(new_A, LA.inv(submatrix))
    print(B)

if __name__ == '__main__':
    find_solution(np.array([[1, 2],[3, 4],[5, 6],[7, 8]]))