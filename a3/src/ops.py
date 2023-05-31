import os
import tvm
from tvm import te


def make_conv1d_cpu_scheduler(M, N):
    A = te.placeholder((M,), name="A")
    W = te.placeholder((N,), name="W")

    padding = N - 1

    padded_A = te.compute(
        (M + 2 * padding,),
        lambda n: tvm.tir.if_then_else(
            tvm.tir.any(n < padding, n >= (M + padding)),
            tvm.tir.const(0.0, "float32"),
            A[n - padding]
        ),
        name="padded_A",
    )

    k = te.reduce_axis((0, N), "k")
    B = te.compute(
        (M + N - 1,),
        lambda n: te.sum(
            padded_A[n + padding - k] * W[k], axis=k
        ),
        name="B",
    )
    s = te.create_schedule(B.op)
    x_out, x_in = s[B].split(B.op.axis[0], factor=16)
    k_out, k_in = s[B].split(k, factor=16)
    s[B].reorder(x_out, k_out, k_in, x_in)
    s[B].unroll(k_in)
    s[B].vectorize(x_in)

    s[s.cache_read(padded_A, 'local', [B])].compute_at(s[B], k_out)
    s[s.cache_read(W, 'local', [B])].compute_at(s[B], k_out)
    
    return s, A, W, B


def make_conv1d_gpu_scheduler(M, N):
    A = te.placeholder((M,), name="A")
    W = te.placeholder((N,), name="W")

    padding = N - 1

    padded_A = te.compute(
        (M + 2 * padding,),
        lambda n: tvm.tir.if_then_else(
            tvm.tir.any(n < padding, n >= (M + padding)),
            tvm.tir.const(0.0, "float32"),
            A[n - padding]
        ),
        name="padded_A",
    )

    k = te.reduce_axis((0, N), "k")
    B = te.compute(
        (M + N - 1,),
        lambda n: te.sum(
            padded_A[n + padding - k] * W[k], axis=k
        ),
        name="B",
    )

    s = te.create_schedule(B.op)
    bx, tx = s[B].split(s[B].op.axis[0], factor=64)
    block = te.thread_axis("blockIdx.x")
    thread = te.thread_axis("threadIdx.x")
    
    s[B].parallel(bx)
    s[B].parallel(tx)
    s[B].bind(bx, block)
    s[B].bind(tx, thread)

    by, ty = s[padded_A].split(s[padded_A].op.axis[0], factor=64)
    s[padded_A].parallel(by)
    s[padded_A].parallel(ty)
    s[padded_A].bind(by, te.thread_axis("blockIdx.y"))
    s[padded_A].bind(ty, te.thread_axis("threadIdx.y"))

    return s, A, W, B


def make_gemm_gpu_scheduler(M, K, N):
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")

    # TVM Matrix Multiplication using TE
    k = te.reduce_axis((0, K), "k")
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    C = te.compute((M, N), lambda x, y: te.sum(A[x, k] * B[k, y], axis=k), name="C")
    # Default schedule
    s = te.create_schedule(C.op)

    # the i-th block is indexed by blockIdx.x.
    # the number of threads in each block is blockDim.x
    # and the i-th thread within a block is indexed by threadIdx.x
    # overall index of a thread can be calculated as
    # 𝑖=blockIdx.x×blockDim.x+threadIdx.x
    y_out, x_out, y_in, x_in = s[C].tile(s[C].op.axis[0], s[C].op.axis[1], 32, 32)
    s[C].parallel(y_out)
    s[C].parallel(x_out)
    s[C].bind(y_out, te.thread_axis("blockIdx.y"))
    s[C].bind(x_out, te.thread_axis("blockIdx.x"))

    k, = s[C].op.reduce_axis
    k_out, k_in = s[C].split(k, factor=32)
    x_in_out, x_in = s[C].split(x_in, nparts=32)
    y_in_out, y_in = s[C].split(y_in, nparts=32)
    
    s[C].reorder(k_out, k_in, x_in, y_in, x_in_out, y_in_out)
    s[C].parallel(x_in_out)
    s[C].parallel(y_in_out)
    s[C].bind(x_in_out, te.thread_axis((0, 32), "threadIdx.x"))
    s[C].bind(y_in_out, te.thread_axis((0, 32), "threadIdx.y"))
    
    s[s.cache_read(A,'local',[C])].compute_at(s[C], k_out)
    s[s.cache_read(B,'local',[C])].compute_at(s[C], k_out)

    return s, A, B, C


def make_dwsp_conv2d_gpu_scheduler(B, C, H, W, K):
    assert K % 2 == 1
    inp = te.placeholder((B, C, H, W), name="A")
    ker = te.placeholder((C, 1, K, K), name="W")

    ki = te.reduce_axis((0, K), 'ki')
    kj = te.reduce_axis((0, K), 'kj')
    ph = (K-1)//2
    pw = ph
    padded_inp = te.compute((B, C, H + K - 1, W + K - 1),
                            lambda *i: tvm.tir.if_then_else(tvm.tir.any(i[-2] < ph, i[-2]>= H + ph, i[-1] < pw, i[-1] >= W + pw),
                            tvm.tir.const(0.0, "float32"),
                            inp[i[:-2] + (i[-2] - ph, i[-1] - pw)]),
                            name='padded_inp')
    out = te.compute((B, C, H, W),
                     lambda b, c, h, w: te.sum(padded_inp[b,c,h+ki,w+kj]*ker[c,0,ki,kj],
                     axis=[ki,kj]),
                     name='out')
    s = te.create_schedule(out.op)

    b_out, c_out, h_out, w_out = s[out].op.axis
    h_out_out, h_out_in = s[out].split(h_out, factor=4)
    w_out_out, w_out_in = s[out].split(w_out, factor=4)

    s[out].reorder(b_out, c_out, h_out_out, w_out_out, h_out_in, w_out_in)
    s[out].parallel(h_out_out)
    s[out].parallel(w_out_out)
    s[out].parallel(h_out_in)
    s[out].parallel(w_out_in)
    s[out].bind(h_out_out, te.thread_axis("blockIdx.x"))
    s[out].bind(w_out_out, te.thread_axis("blockIdx.y"))
    s[out].bind(h_out_in, te.thread_axis("threadIdx.x"))
    s[out].bind(w_out_in, te.thread_axis("threadIdx.y"))

    b_pad, c_pad, h_pad, w_pad = s[padded_inp].op.axis
    h_pad_out, h_pad_in = s[padded_inp].split(h_pad, factor=4)
    w_pad_out, w_pad_in = s[padded_inp].split(w_pad, factor=4)

    s[padded_inp].reorder(b_pad, c_pad, h_pad_out, w_pad_out, h_pad_in, w_pad_in)
    s[padded_inp].parallel(h_pad_out)
    s[padded_inp].parallel(w_pad_out)
    s[padded_inp].parallel(h_pad_in)
    s[padded_inp].parallel(w_pad_in)
    s[padded_inp].bind(h_pad_out, te.thread_axis("blockIdx.z"))
    s[padded_inp].bind(w_pad_out, te.thread_axis("blockIdx.y"))
    s[padded_inp].bind(h_pad_in, te.thread_axis("threadIdx.z"))
    s[padded_inp].bind(w_pad_in, te.thread_axis("threadIdx.y"))

    return s, inp, ker, out
