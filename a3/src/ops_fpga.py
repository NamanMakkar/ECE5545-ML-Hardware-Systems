import os
import vta
from tvm import rpc

def make_conv1d_fpga_scheduler(M,  N):

    env = vta.get_env()

    # We read the Pynq RPC host IP address and port number from the OS environment
    host = os.environ.get("VTA_PYNQ_RPC_HOST", "192.168.2.99")
    port = int(os.environ.get("VTA_PYNQ_RPC_PORT", "9091"))

    # We configure both the bitstream and the runtime system on the Pynq
    # to match the VTA configuration specified by the vta_config.json file.
    if env.TARGET == "pynq":

        # Make sure that TVM was compiled with RPC=1
        assert tvm.module.enabled("rpc")
        remote = rpc.connect(host, port)

        # Reconfigure the JIT runtime
        vta.reconfig_runtime(remote)

        # Program the FPGA with a pre-compiled VTA bitstream.
        # You can program the FPGA with your own custom bitstream
        # by passing the path to the bitstream file instead of None.
        vta.program_fpga(remote, bitstream=None)

    # In simulation mode, host the RPC server locally.
    elif env.TARGET == "sim":
        remote = rpc.LocalSession()

    # TODO: fill-in start
    # TODO: compute scheduler [s] and operator [C] for FPGA
    A = None
    B = None
    C = None
    s = None
    # TODO: fill-in end

    return {
        "scheduler": s,
        "input_A": A,
        "input_B": B,
        "output_C": C,
        "remote": remote,
        "env": env,
        'M': M,
        'N': N
    }


def make_conv1d_fpga_function(scheduler_info):
    """
    Create a function that takes two numpy arrays A, B of
    size (N) and (M) correspondingly, and output a numpy
    array that represents the output C of the function.

    :param scheduler_info:
    :return:
    """
    def func(a_numpy, b_numpy):
        # TODO: fill-in start
        raise NotImplementedError
        # TODO: fill-in end
    return func