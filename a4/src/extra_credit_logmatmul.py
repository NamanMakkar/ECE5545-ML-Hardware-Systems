import torch

def create_lookup_table(size, device):
    lookup_table = torch.zeros(size).to(device)
    for i in range(1, size):
        lookup_table[i] = torch.log2(torch.tensor(1 + 2**(-i))) # stores precomputed values for log(1 + b/a)
    return lookup_table

def log_add_approximation(a, b, lookup_table):
    difference_between_logs = torch.abs(a - b)
    lookup_idx = torch.clamp(torch.round(difference_between_logs*(len(lookup_table) - 1)).long(), 0, len(lookup_table) - 1) # scaling the difference to be within the lookup table
    approximated_sum =  torch.where(a > b, a + lookup_table[lookup_idx], b + lookup_table[lookup_idx]) # max(a,b) + lookup(lookup_idx) => log(a) + log(1 + b/a) => log(a * b)
    return approximated_sum

def logmatmul_approx(A, B, size, device):
    lookup_table = create_lookup_table(size,device)
    lookup_table = lookup_table.to(device)
    N, M = A.shape
    M, K = B.shape

    sign_A = torch.sign(A)
    sign_B = torch.sign(B)

    log_A = torch.log2(torch.abs(A))
    log_B = torch.log2(torch.abs(B))

    sign_A = sign_A.view(N,M,1)
    sign_B = sign_B.view(1,M,K)

    log_A = log_A.view(N,M,1)
    log_B = log_B.view(1,M,K)

    sum_of_logs = log_add_approximation(log_A,log_B,lookup_table)
    product_of_signs = sign_A * sign_B
    product = product_of_signs * (2**sum_of_logs)
    result = torch.sum(product, dim=1)
    return result