import torch
import torch.nn.functional as F


def matmul(A, B, method='naive', **kwargs):
    """
    Multiply two matrices.
    :param A: (N, M) torch tensor.
    :param B: (M, K) torch tensor.
    :param method:
    :return:
        Output matrix with shape (N, K)
    """
    method = method.lower()
    if method in ['naive', 'pytorch', 'torch']:
        return naive(A, B)
    elif method == 'svd':
        return svd(A, B, **kwargs)
    elif method in ['log', 'logmatmul']:
        return logmatmul(A, B, **kwargs)
    else:
        raise ValueError("Invalid [method] value: %s" % method)


def naive(A, B, **kwargs):
    return A @ B


def svd(A, B, rank_A=None, rank_B=None):
    """
    Apply low-rank approximation (SVD) to both matrix A and B with rank rank_A
    and rank_B respectively.
    :param A: (N, M) pytorch tensor
    :param B: (M, K) pytorch tensor
    :param rank_A: None or int. None means use original A matrix.
    :param rank_B: None or int. None means use original B matrix.
    :return: a (N, K) pytorch tensor
    """
    if rank_A is None:
        U_A, S_A, V_A = torch.svd(A)
    else:
        U_A, S_A, V_A = torch.svd(A, some=True)
        U_A = U_A[:,:rank_A]
        S_A = S_A[:rank_A]
        V_A = V_A[:,:rank_A]
    A_new = U_A @ torch.diag(S_A) @ V_A.T
    if rank_B is None:
        U_B, S_B, V_B = torch.svd(B)
    else:
        U_B, S_B, V_B = torch.svd(B, some=True)
        U_B = U_B[:,:rank_B]
        S_B = S_B[:rank_B]
        V_B = V_B[:,:rank_B]
    B_new = U_B @ torch.diag(S_B) @ V_B.T
    result = A_new @ B_new
    return result


def logmatmul(A, B, **kwargs):
    """ TODO: use log multiplication for matrix-matrix multiplication """
    N, M = A.shape # A of shape NxM
    M, K = B.shape # B of shape MxK

    sign_A = torch.sign(A)
    sign_B = torch.sign(B)

    log_A = torch.log2(torch.abs(A))
    log_B = torch.log2(torch.abs(B))

    sign_A = sign_A.view(N, M, 1)
    sign_B = sign_B.view(1, M, K)
    log_A = log_A.view(N, M, 1)
    log_B = log_B.view(1, M, K)

    sum_of_logs = log_A + log_B # We need to take the sum of logs
    product_of_signs = sign_A * sign_B # And the element-wise product of the signs
    product = product_of_signs*(2**sum_of_logs)
    result = torch.sum(product, dim=1)
    return result