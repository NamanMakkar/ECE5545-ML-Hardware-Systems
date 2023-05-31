### ECE 5545 Machine Learning Hardware and Systems

# Assignment 2: Keyword Spotting with MicroControllers

`"We cannot solve our problems with the same thinking we used when we created them.” -- Albert Einstein`

Assigned Date: Monday February 20th    
**Due Date: Monday March 13th**

Total marks = 16/100    
(-2 marks per day late)

----

<p align="center">
  <img src="https://cdn.shopify.com/s/files/1/0506/1689/3647/products/AKX00028_02.unbox_934x700.jpg?v=1615314707"  width="50%" height="50%" class="center" />
</p>

## Overview:
In this lab, you will train and deploy an audio processing neural network on your Arduino Tiny Machine Learning Kit. These are six sections in this assignment as outlined below. In addition to the write-up, make sure to submit all of your code in Github Classroom. Note that there are some automated test cases written to check the functionality of your code and they will be used for grading as well. If you fail a test case but think that your answer is also correct, this is okay, just please provide an explanation in your write-up. Note that the first 4 sections provide a simple walkthrough and some short-answer questions, however, most of the marks will be on sections 5 and 6 where you will implement and investigate quantization and pruning in more depth.

1. [Preprocessing](#preprocessing)
2. [Model Size Estimation](#size)
3. [Training & Analysis](#training)
4. [Model Conversion](#conversion)
5. [Quantization](#quantization)
6. [Pruning](#pruning)

_Note that [Colab Pro](https://colab.research.google.com/signup) or [Colab Pro+](https://colab.research.google.com/signup) are highly recommended for this assignment._

## Part 1: Preprocessing: Audio Recording and Visualization<a name="preprocessing" />
The goal of Part 1 is to become familiar with common audio preprocessing techniques. In this module, you will have an opportunity to record and visualize your own sound bites. Your audio will be plotted in the time domain and frequency domain and as a spectrogram, a mel spectrogram, and lastly a mel frequency cepstral coefficients (MFCC) spectrogram. You will not need to write any code for this part, but do take the time to appreciate how signal transformations enable machine learning algorithms to better extract and utilize features.

### Hand in:
1. Plot the time domain, frequency domain, spectrogram, mel spectrogram, and MFCC spectrogram of your audio.    
2. Comment on why we preprocess input audio sending it through a neural network.     
3. What is the difference between the different spectograms that you plotted? Why does one work better than the other?


## Part 2: Model Size Estimation <a name="size" /> 
MCUs have limited memory and processing power. Before thinking of deploying a model to the MCU we need to check its estimated RAM and flash usage. Furthermore, we will benchmark the speed of the DNN on desktop or server-grade CPU and GPU to compare to the MCU.

### Hand in:
1. Find the estimated flash usage of your DNN. Explain how you arrived at your answer. What percent of your MCU's flash will this model use?    
2. Find the estimated RAM usage of your DNN when operating in batch size = 1. Explain how you arrived at your answer. What percent of your MCU's flash will this model use?    
3. Find the number of FLOPS of your model during a forward pass (or inference). Compare this number to another speech model that you find in the literature. Search with the words "keyword spotting" or "wake word detection" and you should be able to find DNNs that implement similar functionality.   
4. Find the inference runtime of your DNN (batch size = 1) on your Colab cpu and gpu to later compare to the MCU.


## Part 3: Training & Analysis<a name="training" />
Now that we have verified that the model could fit on our MCU, we need to train the model. Please go through the training notebook and understand each step. There is no code that you need to write in this step---it's all there. The result of this notebook is a DNN model checkpoint that you will use in the following notebooks.

### Hand in:
1. Report the accuracy that you get from your model.    
2. Plot curves of the training and validation accuracy during training.    
3. Comment on the speech commands dataset. How many classes of keywords are supported, and how many train/test/validation samples are there?


## Part 4: Model Conversion and Deployment<a name="conversion" />
Our model has been trained using the PyTorch framework, but the MCU only supports TFLite Micro. While converting to a TFLite model, you will quantize the model to reduce its memory and computation. You will use the output of this notebook to deploy the model to your Ardino Nano 33 BLE. Find detailed instructions and notes on how to install the Ardunio SDK and libraries by [following this tutorial](https://docs.google.com/document/d/1lxEl0vGUPKXTwhYTvCpLQMpNNs2OEoT3lgzOfMMPwiU/edit?usp=sharing). You do not need to worry about running the "Blink" example but feel free to do so to verify your installation and board connection. Next, deploy your model onto your TinyML Kit by following [these instructions](arduino_nano_33_ble_tutorial.md).

### Hand in:
1. Profile running time and plot the breakdown between preprocessing, neural network, and post-processing on arduino. Compare to the CPU and GPU numbers you got in part 2. How much slower is the MCU?
2. Record accuracy (out of 10 or more trials) when you try the model with your own voice. Comment on any discrepancy between training accuracy, validation accuracy, and in-field test.
3. [optional] **Extra Credit (1 bonus mark):** Repeat the above question when training with your own keyword. State what keyword you used and how you trained your model with that extra keyword. What accuracy do you achieve for that keyword. Submit a notebook that performs training with this extra keyword.


## Part 5: Quantization-Aware Training<a name="quantization"/>
As we explored in the last two sections, the size of your model is important when dealing with on-device applications. Part 3 implemented a full precision (float 32) model which was quantized after training. In this section, we will use quantization-aware training to try and improve model accuracy during training. Note that we will perform this investigation in PyTorch and not in TFLite.

### Hand in:
1. Code used to finish implementing quantization-aware training (QAT). Provide a very brief explanation of your implementations of the missing functions in notebook 5.
2. Plot accuracy vs. bit-width for:
   *   **post-training quantization** between 2 and 8 bits.
   *   **quantization-aware training** between 2 and 8 bits.
   *   Comment on the impact of post-training quantization versus quantization-aware training.
3. [optional] **Extra Credit (1 bonus mark):** Repeat the above steps with minifloat quantization. You can select the exponent and mantissa bit widths as you see fit. Explain your choices and submit your code.


## Part 6: Pruning<a name="pruning" />
Pruning is another machine learning tactic that can help reduce model size and increase computational efficieny by removing unused parameters in neural networks. Pruning can remove groups of weights in **structured pruning** or individual weights in **unstructured pruning**. For this section, you will implement both structured and unstructured pruning, and measure the impact on accuracy. 

### Hand in:
 1. Code used to implement unstructured and structured pruning, including a fine-tuning step after pruning to regain accuracy. Note that you may use [PyTorch's native pruning library](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html) or implement your own.
 2. For unstructured pruning:
      1. Plot the accuracy vs. number of parameters at different pruning thresholds. Please choose at least 5 pruning thresholds and plot two curves, one with finetuning and one without. You want to aim to plot the "cliff", after which accuracy drops.    
      2. Comment on how we can utilize unstructured pruning to speed up computation. 
      3. What is the difference between L1 norm, L2 norm and L-infinity norm. Which one works best with pruning? 
 3. For structured pruning (channel pruning) plot the following:
      1. Accuracy vs. parameters. Please choose at least 5 pruning thresholds and plot two curves, one with finetuning and one without.    
      2. Accuracy vs. FLOPs. Note that you need to elliminate the pruned channel from the model to compute FLOPS correctly and to perform the following two measurements.
      3. Accuracy vs. runtime on desktop CPU.
      4. Accuracy vs. runtime on MCU (Yes, you need to deploy your pruned model onto the MCU for this step).
