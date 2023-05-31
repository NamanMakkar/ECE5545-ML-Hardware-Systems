### ECE 5545: Machine Learning Hardware and Systems

# Assignment 0: Know Your Design Tools

`"Give me six hours to chop down a tree and I will spend the first four sharpening the axe." -- Abraham Lincoln`

Assigned on: Monday January 23rd    
**Due: Monday February 6th**

Total marks = 2/100

----

## 1. Google Colab and Github Classroom
<p align="center">
  <img src= "https://colab.research.google.com/img/colab_favicon_256px.png" height="200" class="center" />
  <img src= "https://warm-sands-54620.herokuapp.com/assets/pack/original-38e4c80879.png" height="200" class="center" />
</p>

For this and future assignments you will be using GitHub Classroom and Google Colaboratory to get, build, and submit code. The purpose of A0 is to familiarize yourself with these platforms. A0 is also an opportunity for you to ask any questions about the design tools before jumping into A1-A4 which are significantly more work.

### 1.1 GitHub Classroom - Repository

We will distribute starter notebooks and grade your assignments using GitHub Classroom. To complete your assignments you will need:

- [Cornell GitHub Account](https://www.cs.cornell.edu/courses/cs3410/2018sp/resources/username.html). If you want to use a github account that doesn't match your netid, please be sure to email me with your name, netid and github username.
- [GitHub Personal Access Token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token): **Copy your token** to a secure place; you will need it later.
- The email or link from GitHub inviting you to join the class. <br>
  
Once you have sucessfully created your Github account and received the link to A0:
1) Accept the GitHub invitation to the course and follow the props to set up your repository.
2) Open [Google Colab](https://colab.research.google.com/) and sign in using the email address connected to your GitHub account. 
3) Navigate to **File->Upload notebook**. If this is the first time you use Colab, you will likely have to log into your GitHub account using your personal access token. 
4) Search for "ML-HW-SYS" and then select the repository you created in step 1 under "Repository". 
5) Your starter notebook should appear under "Path". Select it and begin familiarizing yourself with Google Colab. 
6) Save a copy of the notebook in your Google Drive and if it was renamed to "copy of a0.ipynb", just rename it back to a0.ipynb.
 
 
### 1.2 Google Colaboratory - Notebook

If this is the first time you use Colab, please go through the following link to familiarize yourself with the platform:
- [Get Started with Google Colaboratory](https://youtu.be/inN8seMm7UI)
- [Google Colab: Basic Features Overview](https://colab.research.google.com/notebooks/basic_features_overview.ipynb) 
- [Uploading and Saving GitHub Notebooks](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb)

 
---

## 2. Notebook Walkthrough and Deliverables

`a0.ipynb` is a simple notebook for MNIST training. It is designed to walk you through simple Colab, Git and Google Drive actions to familiarize you with these platforms for upcoming assignments. Note that we are using Google Colab mainly to have access to GPUs in this course, but if you have access to a powerful CUDA-enabled GPU, you can choose to perform the assignments on your own machine.

Google Colab is an online platform to run Jupyter Notebooks on Google's machines, that optionally have GPUs or TPUs. When you upload a notebook to Colab, your notebook is typically saved to your own Google Drive account that is mounted on the remote machine. All storage on the remote machine is ephemeral, meaning that it will disappear once you log out of the machine. For that reason, we will use Google Drive to save any source files or checkpoints that we need -- a0.ipynb contains commands to mount Google Drive and clone a repository to a Google Drive location. I recommend the following workflow:

- When you open `a0.ipynb` in Colab, save an editable copy straight away to your Google Drive. This will allow you to make your own changes to the notebook. Once you are ready to push a change back to Github, please do so using **File > Save a Copy in Github**, then navigate to your repo.
- Note that pushing changes from your notebook are different than when you want to push changes to source files. You can do the latter using conventional git commands and there is an example of doing that in `a0.ipynb` where you are asked to fix a simple bug then commit the changes to the source file.


We will be using Github Classroom for code submission and grading. A snapshot will be automatically recorded of the `main` branch at the assignment deadline date/time so make sure the code you want to submit is on `main` at the deadline. You can create and work on other branches at any time then merge to `main` before the deadline. Note that Github Classroom may sometimes be configured to run basic tests when you check in code. In a0, there is a simple test that checks the tensor shape of the neural network defined in `model.py`.

Each assignment will also have a writeup to be submitted through gradescope. This is a very important component of the assignment and it should include a detailed answer to any questions in the assignment handout, including diagrams and tables to report results or explain design choices. For this assignment, please include an answer to the following questions in your writeup:

1. Record the time it takes to run 1 training epoch on the CPU and on the GPU. It is good practice to average multiple runs and record the mean and standard deviation.
2. Identify the GPU that you have used and comment on the main features of that GPU. For example, how many cores does it have, what arithmetic precision does it support and at what clock frequency does it run. Comment on why this GPU runs much faster than a CPU for the task of DNN training. On Colab, you can run `!nvidia-smi` in a notebook cell to print information about the current GPU, and you can search the web for other information.
3. Vary the training batch size and plot the time it takes for each training epoch on both the CPU and GPU. Comment on the trends that you see on the plot.
4. Please make sure to push all your code that was used to answer 1-3 above to your Github repo. Please also remember to submit your writeup in Gradescope before the deadline. 
 
**Note that the assignment writeup should be completely self-contained; for example, if you produced a plot in your Colab notebook, you will need to copy it to your writeup. If you are referencing some code that you wrote, you will also need to copy it into your writeup. Treat the writeup like a scientific paper.**
