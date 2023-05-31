
import torch
import torchaudio

import os
import random

import glob
import hashlib
import re
import sys
import tarfile
import math
import urllib
import logging
import sys
import tensorflow as tf 
import numpy as np

from src.constants import *


# If it's available, load the specialized feature generator. If this doesn't
# work, try building with bazel instead of running the Python script directly.
try:
  from tensorflow.lite.experimental.microfrontend.python.ops import audio_microfrontend_op as frontend_op  # pylint:disable=g-import-not-at-top
except ImportError:
  frontend_op = None



def prepare_model_settings(label_count, sample_rate, clip_duration_ms,
                           window_size_ms, window_stride_ms, feature_bin_count,
                           preprocess):
  """Calculates common settings needed for all models.
  Args:
    label_count: How many classes are to be recognized.
    sample_rate: Number of audio samples per second.
    clip_duration_ms: Length of each audio clip to be analyzed.
    window_size_ms: Duration of frequency analysis window.
    window_stride_ms: How far to move in time between frequency windows.
    feature_bin_count: Number of frequency bins to use for analysis.
    preprocess: How the spectrogram is processed to produce features.
  Returns:
    Dictionary containing common settings.
  Raises:
    ValueError: If the preprocessing mode isn't recognized.
  """
  desired_samples = int(sample_rate * clip_duration_ms / 1000)
  window_size_samples = int(sample_rate * window_size_ms / 1000)
  window_stride_samples = int(sample_rate * window_stride_ms / 1000)
  length_minus_window = (desired_samples - window_size_samples)
  if length_minus_window < 0:
    spectrogram_length = 0
  else:
    spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
  if preprocess == 'micro':
    average_window_width = -1
    fingerprint_width = feature_bin_count
  else:
    raise ValueError('Unknown preprocess mode "%s" (should be "micro")' % (preprocess))
  fingerprint_size = fingerprint_width * spectrogram_length
  return {
      'desired_samples': desired_samples,
      'window_size_samples': window_size_samples,
      'window_stride_samples': window_stride_samples,
      'spectrogram_length': spectrogram_length,
      'fingerprint_width': fingerprint_width,
      'fingerprint_size': fingerprint_size,
      'label_count': label_count,
      'sample_rate': sample_rate,
      'preprocess': preprocess,
      'average_window_width': average_window_width,
  }


def prepare_words_list(wanted_words):
  """Prepends common tokens to the custom word list.
  Args:
    wanted_words: List of strings containing the custom words.
  Returns:
    List with the standard silence and unknown tokens added.
  """
  return [SILENCE_LABEL, UNKNOWN_WORD_LABEL] + wanted_words


def which_set(filename, validation_percentage, testing_percentage):
  """Determines which data partition the file should belong to.
  We want to keep files in the same training, validation, or testing sets even
  if new ones are added over time. This makes it less likely that testing
  samples will accidentally be reused in training when long runs are restarted
  for example. To keep this stability, a hash of the filename is taken and used
  to determine which set it should belong to. This determination only depends on
  the name and the set proportions, so it won't change as other files are added.
  It's also useful to associate particular files as related (for example words
  spoken by the same person), so anything after '_nohash_' in a filename is
  ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
  'bobby_nohash_1.wav' are always in the same set, for example.
  Args:
    filename: File path of the data sample.
    validation_percentage: How much of the data set to use for validation.
    testing_percentage: How much of the data set to use for testing.
  Returns:
    String, one of 'training', 'validation', or 'testing'.
  """
  base_name = os.path.basename(filename)
  # We want to ignore anything after '_nohash_' in the file name when
  # deciding which set to put a wav in, so the data set creator has a way of
  # grouping wavs that are close variations of each other.
  hash_name = re.sub(r'_nohash_.*$', '', base_name)
  # eg. hash_name: '0a2b400e'
  
  # This looks a bit magical, but we need to decide whether this file should
  # go into the training, testing, or validation sets, and we want to keep
  # existing files in the same set even if more files are subsequently
  # added.
  # To do that, we need a stable way of deciding based on just the file name
  # itself, so we do a hash of that and then use that to generate a
  # probability value that we use to assign it.
  hash_name_hashed = hashlib.sha1(bytes(hash_name, 'utf-8')).hexdigest()
  percentage_hash = ((int(hash_name_hashed, 16) %
                      (MAX_NUM_WAVS_PER_CLASS + 1)) *
                     (100.0 / MAX_NUM_WAVS_PER_CLASS))
  if percentage_hash < validation_percentage:
    result = 'validation'
  elif percentage_hash < (testing_percentage + validation_percentage):
    result = 'testing'
  else:
    result = 'training'
  return result


class AudioProcessor(object):
  """Handles loading, partitioning, and preparing audio training data."""
  def __init__(self, data_dir=DATASET_DIR, data_url=DATA_URL):
      self.data_dir = data_dir
      self.num_labels = len(prepare_words_list(WANTED_WORDS))
      self.model_settings = prepare_model_settings(label_count=self.num_labels, \
          sample_rate=SAMPLE_RATE, clip_duration_ms=CLIP_DURATION_MS, window_size_ms=WINDOW_SIZE_MS, \
              window_stride_ms=WINDOW_STRIDE, feature_bin_count=FEATURE_BIN_COUNT, preprocess=PREPROCESS)
      self.maybe_download_and_extract_dataset(data_url=data_url)
      self.prepare_data_index(silence_percentage=SILENT_PERCENTAGE, unknown_percentage=UNKNOWN_PERCENTAGE, wanted_words=WANTED_WORDS,\
         validation_percentage=VALIDATION_PERCENTAGE, testing_percentage=TESTING_PERCENTAGE)
      self.prepare_background_data()
      self.output = None

  def set_size(self, mode):
    """Calculates the number of samples in the dataset partition.
    Args:
      mode: Which partition, must be 'training', 'validation', or 'testing'.
    Returns:
      Number of samples in the partition.
    """
    return len(self.data_index[mode])

  def get_data(self, how_many, offset, background_frequency,
               background_volume_range, time_shift, mode):
    """Gather samples from the data set, applying transformations as needed.
    When the mode is 'training', a random selection of samples will be returned,
    otherwise the first N clips in the partition will be used. This ensures that
    validation always uses the same samples, reducing noise in the metrics.
    Args:
      how_many: Desired number of samples to return. -1 means the entire
        contents of this partition.
      offset: Where to start when fetching deterministically.
      model_settings: Information about the current model being trained.
      background_frequency: How many clips will have background noise, 0.0 to
        1.0.
      background_volume_range: How loud the background noise will be.
      time_shift: How much to randomly shift the clips by in time.
      mode: Which partition to use, must be 'training', 'validation', or
        'testing'.
    Returns:
      List of sample data for the transformed samples, and list of label indexes
    Raises:
      ValueError: If background samples are too short.
    """
    # Pick one of the partitions to choose samples from.
    candidates = self.data_index[mode]
    if how_many == -1:
      sample_count = len(candidates)
    else:
      sample_count = max(0, min(how_many, len(candidates) - offset))
    # Data and labels will be populated and returned.
    data = torch.zeros((sample_count, self.model_settings['fingerprint_size'])) # shape: [sample_count,1960]
    labels = np.zeros(sample_count)
    desired_samples = self.model_settings['desired_samples']
    use_background = self.background_data and (mode == 'training')
    pick_deterministically = (mode != 'training')
    # Use the processing graph we created earlier to repeatedly to generate the
    # final output sample data we'll use in training.
    for i in range(offset, offset + sample_count):
      # Pick which audio sample to use.
      if how_many == -1 or pick_deterministically:
        sample_index = i
      else:
        sample_index = torch.randint(len(candidates), [1])
      sample = candidates[sample_index]
      # If we're time shifting, set up the offset for this sample.
      if time_shift > 0:
        time_shift_amount = torch.randint(-int(time_shift), int(time_shift), [1])
      else:
        time_shift_amount = 0
      if time_shift_amount > 0: 
        time_shift_padding = (time_shift_amount, 0)
        time_shift_offset=0
      else:
        time_shift_padding = (0, -time_shift_amount)
        time_shift_offset=-time_shift_amount

      wav_filename= sample['file']
      time_shift_padding= time_shift_padding
      time_shift_offset= time_shift_offset

      # Choose a section of background noise to mix in.
      if use_background or sample['label'] == SILENCE_LABEL:
        background_index = torch.randint(len(self.background_data), [1])
        background_samples, background_sample_rate = self.background_data[background_index]
        if background_samples.numel() <= self.model_settings['desired_samples']:
          raise ValueError(
              'Background sample is too short! Need more than %d'
              ' samples but only %d were found' %
              (self.model_settings['desired_samples'], background_samples.numel()))
        background_offset = torch.randint(
            0, background_samples.numel() - self.model_settings['desired_samples'], [1])
        background_clipped = background_samples[:, background_offset:
            background_offset + desired_samples] #[1,16000]
        #background_reshaped = torch.reshape(background_clipped, [desired_samples, 1])
        background_reshaped = background_clipped
        if sample['label'] == SILENCE_LABEL:
          background_volume = torch.rand([1])
        elif torch.rand([1]) < background_frequency:
          background_volume = torch.rand([1]) * background_volume_range
        else:
          background_volume = 0
      else:
        background_reshaped = torch.zeros([1, desired_samples])
        background_volume = 0
      background_data = background_reshaped
      background_volume = background_volume
      # If we want silence, mute out the main sample but leave the background.
      if sample['label'] == SILENCE_LABEL:
        foreground_volume = 0
      else:
        foreground_volume = 1

      # run the graph
      self.prepare_processing_graph(wav_filename, foreground_volume, time_shift_offset, \
          time_shift_padding, background_data, background_volume)
      data_tensor = self.output
      data[i - offset, :] = data_tensor.flatten()
      label_index = self.word_to_index[sample['label']]
      labels[i - offset] = label_index
    return data, torch.tensor(labels)

  def get_data_from_file(self, sample, background_frequency,
               background_volume_range, time_shift, mode):
    """
    Args:
      sample: a dictionary containing a label and a wav file
      background_frequency: How many clips will have background noise, 0.0 to
        1.0.
      background_volume_range: How loud the background noise will be.
      time_shift: How much to randomly shift the clips by in time.
      mode: Which partition to use, must be 'training', 'validation', or
        'testing'. Add background only if mode is 'training'
    Returns:
      List of sample data for the transformed samples, and list of label indexes
    Raises:
      ValueError: If background samples are too short.
    """    

    sample_count = 1
    # Data and labels will be populated and returned.
    data = torch.zeros((sample_count, self.model_settings['fingerprint_size'])) 
    labels = np.zeros(sample_count)
    desired_samples = self.model_settings['desired_samples']
    use_background = self.background_data and (mode == 'training')
    
      # If we're time shifting, set up the offset for this sample.
    if time_shift > 0:
      time_shift_amount = torch.randint(-time_shift, time_shift, [1])
    else:
      time_shift_amount = 0
    if time_shift_amount > 0:
      time_shift_padding = (time_shift_amount, 0)
      time_shift_offset=0
    else:
      time_shift_padding = (0, -time_shift_amount)
      time_shift_offset=-time_shift_amount

    wav_filename= sample['file']
    time_shift_padding= time_shift_padding
    time_shift_offset= time_shift_offset

    # Choose a section of background noise to mix in.
    if use_background or sample['label'] == SILENCE_LABEL:
      background_index = torch.randint(len(self.background_data), [1])
      background_samples, background_sample_rate = self.background_data[background_index]
      if background_samples.numel() <= self.model_settings['desired_samples']:
        raise ValueError(
            'Background sample is too short! Need more than %d'
            ' samples but only %d were found' %
            (self.model_settings['desired_samples'], background_samples.numel()))
      background_offset = torch.randint(
          0, background_samples.numel() - self.model_settings['desired_samples'], [1])
      background_clipped = background_samples[:, background_offset:
          background_offset + desired_samples] #[1,16000]

      background_reshaped = background_clipped
      if sample['label'] == SILENCE_LABEL:
        background_volume = torch.rand([1])
      elif torch.rand([1]) < background_frequency:
        background_volume = torch.rand([1]) * background_volume_range
      else:
        background_volume = 0
    else:
      # do not add background 
      background_reshaped = torch.zeros([1, desired_samples])
      background_volume = 0

    background_data = background_reshaped

    # If we want silence, mute out the main sample but leave the background.
    if sample['label'] == SILENCE_LABEL:
      foreground_volume = 0
    else:
      foreground_volume = 1

    # run the graph
    self.prepare_processing_graph(wav_filename, foreground_volume, time_shift_offset, \
        time_shift_padding, background_data, background_volume)
    data_tensor = self.output
    data[0, :] = data_tensor.flatten()
    label_index = self.word_to_index[sample['label']]
    labels[0] = label_index
    return data, torch.tensor(labels)

  def get_features_for_wav(self, wav_filename):
    """Applies the feature transformation process to the input_wav.
    Runs the feature generation process (generally producing a spectrogram from
    the input samples) on the WAV file. This can be useful for testing and
    verifying implementations being run on other platforms.
    Args:
      wav_filename: The path to the input audio file.
      model_settings: Information about the current model being trained.
      sess: TensorFlow session that was active when processor was created.
    Returns:
      Numpy data array containing the generated features.
    """
    desired_samples = self.model_settings['desired_samples']
    time_shift_padding= (0,0,0,0)
    time_shift_offset= 0
    background_data= np.zeros([desired_samples, 1])
    background_volume= 0
    foreground_volume= 1

    # Run the graph to produce the output audio.
    self.prepare_processing_graph(wav_filename, foreground_volume, time_shift_offset, \
        time_shift_padding, background_data, background_volume)

    return self.output

  def maybe_download_and_extract_dataset(self, data_url):
    """Download and extract data set tar file.
    If the data set we're using doesn't already exist, this function
    downloads it from the TensorFlow.org website and unpacks it into a
    directory.
    If the data_url is none, don't download anything and expect the data
    directory to contain the correct files already.
    Args:
      data_url: Web location of the tar file containing the data set.
      dest_directory: File path to extract data to.
    """
    dest_directory = self.data_dir 
    if not data_url:
      return
    if not os.path.isdir(dest_directory):
      os.makedirs(dest_directory)
    filename = data_url.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.isfile(filepath):
      def _progress(count, block_size, total_size):
        sys.stdout.write(
            '\r>> Downloading %s %.1f%%' %
            (filename, float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()

      try:
        filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
      except:
        logging.error(
            'Failed to download URL: {0} to folder: {1}. Please make sure you '
            'have enough free space and an internet connection'.format(
                data_url, filepath))
        raise
      print()
      statinfo = os.stat(filepath)
      logging.info(
          'Successfully downloaded {0} ({1} bytes)'.format(
              filename, statinfo.st_size))
      tarfile.open(filepath, 'r:gz').extractall(dest_directory)

  def prepare_data_index(self, silence_percentage, unknown_percentage,
                         wanted_words, validation_percentage,
                         testing_percentage):
    """Prepares a list of the samples organized by set and label.
    The training loop needs a list of all the available data, organized by
    which partition it should belong to, and with ground truth labels attached.
    This function analyzes the folders below the `data_dir`, figures out the
    right
    labels for each file based on the name of the subdirectory it belongs to,
    and uses a stable hash to assign it to a data set partition.
    Args:
      silence_percentage: How much of the resulting data should be background.
      unknown_percentage: How much should be audio outside the wanted classes.
      wanted_words: Labels of the classes we want to be able to recognize.
      validation_percentage: How much of the data set to use for validation.
      testing_percentage: How much of the data set to use for testing.
    Returns:
      Dictionary containing a list of file information for each set partition,
      and a lookup map for each class to determine its numeric index.
    Raises:
      Exception: If expected files are not found.
    """
    # Make sure the shuffling and picking of unknowns is deterministic.
    random.seed(RANDOM_SEED)
    wanted_words_index = {}
    for index, wanted_word in enumerate(wanted_words):
      wanted_words_index[wanted_word] = index + 2
    self.data_index = {'validation': [], 'testing': [], 'training': []}
    unknown_index = {'validation': [], 'testing': [], 'training': []}
    all_words = {}
    # Look through all the subfolders to find audio samples
    search_path = os.path.join(self.data_dir, '*', '*.wav')
    for wav_path in glob.glob(search_path):
      _, word = os.path.split(os.path.dirname(wav_path))
      word = word.lower()
      # Treat the '_background_noise_' folder as a special case, since we expect
      # it to contain long audio samples we mix in to improve training.
      if word == BACKGROUND_NOISE_DIR_NAME:
        continue
      all_words[word] = True
      set_index = which_set(wav_path, validation_percentage, testing_percentage)
      # If it's a known class, store its detail, otherwise add it to the list
      # we'll use to train the unknown label.
      if word in wanted_words_index:
        self.data_index[set_index].append({'label': word, 'file': wav_path})
      else:
        unknown_index[set_index].append({'label': word, 'file': wav_path})
    if not all_words:
      raise Exception('No .wavs found at ' + search_path)
    for index, wanted_word in enumerate(wanted_words):
      if wanted_word not in all_words:
        raise Exception('Expected to find ' + wanted_word +
                        ' in labels but only found ' +
                        ', '.join(all_words.keys()))
    # We need an arbitrary file to load as the input for the silence samples.
    # It's multiplied by zero later, so the content doesn't matter.
    silence_wav_path = self.data_index['training'][0]['file']
    for set_index in ['validation', 'testing', 'training']:
      set_size = len(self.data_index[set_index])
      silence_size = int(math.ceil(set_size * silence_percentage / 100))
      for _ in range(silence_size):
        self.data_index[set_index].append({
            'label': SILENCE_LABEL,
            'file': silence_wav_path
        })
      # Pick some unknowns to add to each partition of the data set.
      random.shuffle(unknown_index[set_index])
      unknown_size = int(math.ceil(set_size * unknown_percentage / 100))
      self.data_index[set_index].extend(unknown_index[set_index][:unknown_size])
    # Make sure the ordering is random.
    for set_index in ['validation', 'testing', 'training']:
      random.shuffle(self.data_index[set_index])
    # Prepare the rest of the result data structure.
    self.words_list = prepare_words_list(wanted_words)
    self.word_to_index = {}
    for word in all_words:
      if word in wanted_words_index:
        self.word_to_index[word] = wanted_words_index[word]
      else:
        self.word_to_index[word] = UNKNOWN_WORD_INDEX
    self.word_to_index[SILENCE_LABEL] = SILENCE_INDEX

  def prepare_background_data(self):
    """Searches a folder for background noise audio, and loads it into memory.
    It's expected that the background audio samples will be in a subdirectory
    named '_background_noise_' inside the 'data_dir' folder, as .wavs that match
    the sample rate of the training data, but can be much longer in duration.
    If the '_background_noise_' folder doesn't exist at all, this isn't an
    error, it's just taken to mean that no background noise augmentation should
    be used. If the folder does exist, but it's empty, that's treated as an
    error.
    Returns:
      List of raw PCM-encoded audio samples of background noise.
    Raises:
      Exception: If files aren't found in the folder.
    """
    self.background_data = []
    background_dir = os.path.join(self.data_dir, BACKGROUND_NOISE_DIR_NAME)
    if not os.path.isdir(background_dir):
      return self.background_data
    
    search_path = os.path.join(self.data_dir, BACKGROUND_NOISE_DIR_NAME, '*.wav')
  
    for wav_path in glob.glob(search_path):
      wav_data = torchaudio.load(wav_path)
      self.background_data.append(wav_data)
      if not self.background_data:
        raise Exception('No background wav files were found in ' + search_path)

  def prepare_processing_graph(self, wav_filename, foreground_volume, time_shift_offset, \
      time_shift_padding, background_data, background_volume):
    """Builds a TensorFlow graph to apply the input distortions.
    Creates a graph that loads a WAVE file, decodes it, scales the volume,
    shifts it in time, adds in background noise, calculates a spectrogram, and
    then builds an MFCC fingerprint from that.
    This must be called with an active TensorFlow session running, and it
    creates multiple placeholder inputs, and one output:
      - wav_filename_placeholder_: Filename of the WAV to load.
      - foreground_volume_placeholder_: How loud the main clip should be.
      - time_shift_padding_placeholder_: Where to pad the clip.
      - time_shift_offset_placeholder_: How much to move the clip in time.
      - background_data_placeholder_: PCM sample data for background noise.
      - background_volume_placeholder_: Loudness of mixed-in background.
      - output_: Output 2D fingerprint of processed audio.
    Args:
      model_settings: Information about the current model being trained.
      summaries_dir: Path to save training summary information to.
    Raises:
      ValueError: If the preprocessing mode isn't recognized.
      Exception: If the preprocessor wasn't compiled in.
    """
    # loads and decode a WAVE file
    desired_samples = self.model_settings['desired_samples']
    waveform, sample_rate = torchaudio.load(wav_filename, frame_offset=0, num_frames=desired_samples)
    total_num_samples=waveform.shape[1]
    if total_num_samples < desired_samples:
      waveform = torch.nn.functional.pad(waveform, (0,desired_samples-total_num_samples))

    # scale the volume
    foreground_volume = foreground_volume
    waveform = torch.multiply(waveform, foreground_volume)

    # Padding
    num_frames = desired_samples
    frame_offset = time_shift_offset
    pad = time_shift_padding
    waveform = torch.nn.functional.pad(waveform, pad, mode='constant', value=0)

    sliced_waveform = waveform[:, frame_offset:frame_offset+num_frames]

    # Mix in background noise.
    background_data = background_data
    background_volume = background_volume
    background = torch.add(torch.multiply(torch.tensor(background_data.numpy()), background_volume), sliced_waveform)
    background_clamp = torch.clamp(background, -1.0, 1.0)


    if self.model_settings['preprocess'] == 'micro':
        background_clamp_tf = tf.convert_to_tensor(background_clamp.numpy())
        if not frontend_op:
          raise Exception(
              'Micro frontend op is currently not available when running'
              ' TensorFlow directly from Python, you need to build and run'
              ' through Bazel')
        sample_rate = self.model_settings['sample_rate']
        window_size_ms = (self.model_settings['window_size_samples'] *
                          1000) / sample_rate
        window_step_ms = (self.model_settings['window_stride_samples'] *
                          1000) / sample_rate
        int16_input = tf.cast(tf.multiply(background_clamp_tf, 32768), tf.int16)
        micro_frontend = frontend_op.audio_microfrontend(
            int16_input,
            sample_rate=sample_rate,
            window_size=window_size_ms,
            window_step=window_step_ms,
            num_channels=self.model_settings['fingerprint_width'],
            out_scale=1,
            out_type=tf.float32)
        tf_mult = tf.multiply(micro_frontend, (10.0 / 256.0))
        self.output = torch.tensor(tf_mult.numpy(), dtype= torch.float32)
      
    else:
        raise ValueError('Unknown preprocess mode "%s" (should be "micro")' %
                         (self.model_settings['preprocess']))


