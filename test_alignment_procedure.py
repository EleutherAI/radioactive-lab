#from numpy.random import default_rng # Doesnt work for me
import numpy as np

if __name__ == '__main__':
    dimensions = 2
    examples = 5

    #rng = default_rng()
    values_basis1 = np.random.random((examples, dimensions))
    M = np.random.random((dimensions, dimensions))
    values_basis2 = np.matmul(values_basis1, M)

    print("values_basis1")
    print(values_basis1)
    print("values_basis2")
    print(values_basis2)

    M_hat, residuals, rank, s = np.linalg.lstsq(values_basis1, values_basis2)
    print("Norm of residual: %.4e" % np.linalg.norm(np.dot(values_basis1, M_hat) - values_basis2)**2)
    values_basis2_hat = np.matmul(values_basis1, M_hat)

    print("M")
    print(M)
    print("M_hat")
    print(M_hat)
    print("values_basis2_hat")
    print(values_basis2_hat)