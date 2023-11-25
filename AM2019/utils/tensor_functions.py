import torch
from loguru import logger

# modified from the code provided by ChatGPT
def randomized_svd_batch(A_batch, k, p=5, num_iterations=1):
    """
    Compute the first k singular values of a batch of matrices using randomized SVD.
    I change this based on alg.2 in https://gregorygundersen.com/blog/2019/01/17/randomized-svd/
    Actually based on paper: https://arxiv.org/pdf/0909.4061.pdf

    Parameters:
    - A_batch: torch.Tensor, input batch of matrices with dimensions (batch_size, m, n)
    - k: int, number of singular values to compute
    - num_iterations: int, number of iterations for power iteration

    Returns:
    - U_batch: torch.Tensor, left singular vectors for each matrix in the batch
    - S_batch: torch.Tensor, singular values for each matrix in the batch
    - V_batch: torch.Tensor, right singular vectors for each matrix in the batch
    """
    device = A_batch.device
    batch_size, m, n = A_batch.shape
    # logger.debug(f"A_batch max: {torch.max(A_batch)}, min: {torch.min(A_batch)}")
    sketch_size = min(k+p, n)  # Adjust sketch size based on problem size

    # Generate a random Gaussian matrix for each matrix in the batch
    Omega_batch = torch.randn((batch_size, n, sketch_size), device=device)

    # Form the sketched matrices for each matrix in the batch
    Y_batch = torch.matmul(A_batch, Omega_batch)

    # Perform power iteration for each matrix in the batch
    for _ in range(num_iterations):
        Y_batch = torch.matmul(A_batch, torch.matmul(A_batch.transpose(1, 2), Y_batch))

    # logger.debug(f"Y_batch max: {torch.max(Y_batch)}, min: {torch.min(Y_batch)}")
    # QR decomposition of the sketched matrices for each matrix in the batch
    Q_batch, _ = torch.linalg.qr(Y_batch)
    # logger.debug(f"Q_batch max: {torch.max(Q_batch)}, min: {torch.min(Q_batch)}")

    # Compute the SVD of the sketched matrices for each matrix in the batch
    B_batch = torch.matmul(Q_batch.transpose(1, 2), A_batch)
    # logger.debug(f"B_batch max: {torch.max(B_batch)}, min: {torch.min(B_batch)}")
    big_M = 1e4 # FIXME
    B_batch = torch.clamp(B_batch, -big_M, big_M)

    U_batch, S_batch, Vt_batch = torch.linalg.svd(B_batch)
    V_batch = Vt_batch.transpose(1, 2)
    U_batch = torch.matmul(Q_batch, U_batch)

    # Return the first k singular values and vectors for each matrix in the batch
    return U_batch[:, :, :k], S_batch[:, :k], V_batch[:, :, :k]



def compute_in_batches(f, calc_batch_size, *args, n=None):
    """
    Computes memory heavy function f(*args) in batches
    :param n: the total number of elements, optional if it cannot be determined as args[0].size(0)
    :param f: The function that is computed, should take only tensors as arguments and return tensor or tuple of tensors
    :param calc_batch_size: The batch size to use when computing this function
    :param args: Tensor arguments with equally sized first batch dimension
    :return: f(*args), this should be one or multiple tensors with equally sized first batch dimension
    """
    if n is None:
        n = args[0].size(0)
    n_batches = (n + calc_batch_size - 1) // calc_batch_size  # ceil
    if n_batches == 1:
        return f(*args)

    # Run all batches
    # all_res = [f(*batch_args) for batch_args in zip(*[torch.chunk(arg, n_batches) for arg in args])]
    # We do not use torch.chunk such that it also works for other classes that support slicing
    all_res = [f(*(arg[i * calc_batch_size:(i + 1) * calc_batch_size] for arg in args)) for i in range(n_batches)]

    # Allow for functions that return None
    def safe_cat(chunks, dim=0):
        if chunks[0] is None:
            assert all(chunk is None for chunk in chunks)
            return None
        return torch.cat(chunks, dim)

    # Depending on whether the function returned a tuple we need to concatenate each element or only the result
    if isinstance(all_res[0], tuple):
        return tuple(safe_cat(res_chunks, 0) for res_chunks in zip(*all_res))
    return safe_cat(all_res, 0)
