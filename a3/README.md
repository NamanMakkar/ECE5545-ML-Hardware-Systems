### ECE 5545 Machine Learning Hardware and Systems

# Assignment 3: Compiling DNNs with TVM

`"I'd rather have a search engine or a compiler on a deserted island than a game." -- John Carmack (CTO of Oculus VR)`

Assigned on: Wednesday March 15th    
**Due: Wednesday April 12th**

Total marks = 16/100    
(-2 marks per day late)

----

<p align="center">
  <img src= "https://tvm.apache.org/images/main/tvm-stack.png" height="300" class="center" />
</p>

## Overview

In this assignment, we will use [TVM](https://tvm.apache.org/) to optimize DNN primitives on CPU, GPU and FPGA devices. TVM is a domain-specific compiler for machine learning in which the program computation specification and schedule are decoupled. This means that we specify the computation separately from the optimizations that actually allow the code to run efficiently on hardware.

## Implementation

You will optimize the following primitives:

1. 1D convolution on CPU^.
1. 1D convolution on GPU.
1. 1D convolution on FPGA.
1. GEMM on GPU^.
1. 2D Depthwise Separable Convolution on GPU.

^ _A Baseline implementation is provided for the 1D convolution on the CPU, and GEMM on GPU; you are required to optimize these implementations. For other problems, you are required to write the baseline implementation **and** optimize them._

For each of those primitives, there is a functionality test that will run to ensure that your optimized code is still functional. Additionally, we will benchmark all submissions to find the fastest solution. Higher marks will be awarded to faster solutions!

You are expected to perform at least three distinct optimizations for each operation, but you are encouraged to do more. There are plenty of examples on the TVM website to guide you on the _kind_ of optimizations that work on each device. For example, this [GEMM on CPU optimization](https://tvm.apache.org/docs/tutorial/tensor_expr_get_started.html#example-2-manually-optimizing-matrix-multiplication-with-te) showcases 6 distinct optimizations. It will be very useful for you to walk through these optimizations and understand them before diving into your solution for the questions above. Other useful examples include this [2D Conv Optimization on GPU](https://tvm.apache.org/docs/how_to/optimize_operators/opt_conv_cuda.html?highlight=gpu%20conv%20optimization) and [GEMM on FPGA (VTA)](https://tvm.apache.org/docs/topic/vta/tutorials/matrix_multiply.html).

You will use TVM's [schedule primitives](https://tvm.apache.org/docs/how_to/work_with_schedules/schedule_primitives.html) to perform the optimizations. You are not allowed to use AutoTVM, although you may do so for comparison [1 mark extra credit].

In your code submission, make sure to include all of your optimizations, ideally, organized into different functions. 

## Write-up

* Your report should include tables or charts to showcase the **runtime** and **speedup (relative to unoptimized TVM  implementation)** of each DNN primitive. For the CPU/GPU it would also be useful to include the speedup/slowdown relative to the provided torch/numpy reference. 
* For each distinct optimization that you performed, please explain it in some detail: why you performed this optimization? And what impact does it have on the hardware execution? You are encouraged to use figures for explanation, similar to the examples on the TVM website. It may also be useful to show the effect on the TVM IR from the `tvm.lower` function as this highlights the effect of optimizations--pseudo-code (see Figure 5.8 in textbook) also works well for this purpose. 
* [Optional] You may find it useful to plot space-time diagrams of your memory accesses (see Figure 5.9 in our textbook). These may aid your choice of tiling and vectorization optimizations. It is not required to include these plots in your report but you may choose to do so if it helps.
* Note: If you tried an interesting optimization but it did not improve your runtime, you can also describe it in your writeup and report on the slowdown and potential reasons why it didn't work. You will be awarded partial marks for this if you provide a meaningful analysis.
* Please do **not** include your source code in the report unless you are referencing it in the text--your code will be assessed separately through your Github submission.
