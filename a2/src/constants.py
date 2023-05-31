import os

# Constants for audio process during Quantization and Evaluation
# This values are used for determining correct settings for feature generator used in data_proc.py
# These values are used in the original tensorflow speech_command example
# DO NOT MODIFY 
SAMPLE_RATE = 16000
CLIP_DURATION_MS = 1000
WINDOW_SIZE_MS = 30.0
FEATURE_BIN_COUNT = 40
BACKGROUND_FREQUENCY = 0.8
BACKGROUND_VOLUME_RANGE = 0.1
TIME_SHIFT_MS = 100.0
PREPROCESS = 'micro'
WINDOW_STRIDE = 20
TIME_SHIFT_SAMPLE = int((TIME_SHIFT_MS * SAMPLE_RATE) / 1000)


# MAX_NUM_WAVS_PER_CLASS is an arbitrary value, 
# which is a metric that determines the proportion of subsequent hash value generation
# This constant is used when we use hashing to decide whether a wav file should
# go into the training, testing, or validating set
# This specific value is used in the original tensorflow speech_command example
# DO NOT MODIFY
MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M

# Labels and indices used for silence and unkown words
SILENCE_LABEL = '_silence_'
SILENCE_INDEX = 0
UNKNOWN_WORD_LABEL = '_unknown_'
UNKNOWN_WORD_INDEX = 1

# The name of the background noise directory inside the dataset directory 
# DO NOT MODIFY if you are using the default DATA_URL
# MAKE SURE this is the correct directory name for background noises 
# if you are using a different dataset
BACKGROUND_NOISE_DIR_NAME = '_background_noise_'

# RANDOM_SEED used in data_proc.py to make sure the shuffling and picking of unknowns is deterministic
# Changing this value might change the efficiency and accuracy of training
RANDOM_SEED = 59185

# Calculate the percentage of 'silence' and 'unknown' training samples required
# to ensure that we have equal number of samples for each label.
# Change this value if you want to train the model to identify different words
# Choosing 2-4 words will have better training results
WANTED_WORDS = ["yes", "no", "dog"]

# Calculate percentage of training samples so each class will have approcimately 
# equal number of samples for training, testing, and validating
# DO NOT MODIFY
number_of_labels = WANTED_WORDS.count(',') + 1
number_of_total_labels = number_of_labels + 2 # for 'silence' and 'unknown' label
equal_percentage_of_samples = int(100.0/(number_of_total_labels))
SILENT_PERCENTAGE = equal_percentage_of_samples
UNKNOWN_PERCENTAGE = equal_percentage_of_samples

# Percentage of labels for testing and validating
# Modifying these values might change the efficiency and accuracy of training
VALIDATION_PERCENTAGE = 10
TESTING_PERCENTAGE = 10

# Batch size used for training
# Modifying this value might change the efficiency and accuracy of training
BATCH_SIZE = 100

# url for the dataset we used for training in this lab
DATA_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz'

#GitHub directory path
# GIT_DIR = '/gdrive/MyDrive/ece5545/a2'
GIT_DIR = os.getcwd()

# Google Drive directory 
# Sets up folder in student's Google Drive
# DRIVE_DIR = "/gdrive/MyDrive/ece5545/a2"
DRIVE_DIR = os.getcwd()

# The directory where the dataset will be downloaded
# You can change this value, but please keep it in the same directory with other files from this lab
DATASET_DIR = DRIVE_DIR + '/dataset'

# Model dirs
# Pathes to directories where trained models are saved
# You can change names of these directories, but keep them simple and obvious 
MODEL_DIR = DRIVE_DIR + "/models"
TORCH_DIR = MODEL_DIR + "/torch_models"
ONNX_DIR = MODEL_DIR + "/onnx_models"
TF_DIR = MODEL_DIR + "/tf_models"
TFLITE_DIR = MODEL_DIR + "/tflite_models"
MICRO_DIR = MODEL_DIR + "/micro_models"


# Create model folders based on the pathes defined above
# The following codes will be executed when constants.py are imported in other files
# DO NOT MODIFY
import os
for dir in [MODEL_DIR, TORCH_DIR, ONNX_DIR, TF_DIR, TFLITE_DIR, MICRO_DIR]:
    if not os.path.isdir(dir):
        os.makedirs(dir)
print('Model folders are created, \n' + \
    f'PyTorch models will be saved in {TORCH_DIR}, \n' + \
    f'ONNX models will be saved in {ONNX_DIR}, \n' + \
    f'TensorFlow Saved Models will be saved in {TF_DIR}, \n' + \
    f'TensorFlow Lite models will be saved in {TFLITE_DIR}, \n' + \
    f'TensorFlow Lite Micro models will be saved in {MICRO_DIR}.' )
