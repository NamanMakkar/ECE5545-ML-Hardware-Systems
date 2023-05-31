import torch
from src.constants import *


# Define pad and collate functions
def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch


def make_collate_fn(audio_processor, mode):
    def collate_fn_(batch):
        tensors, targets = [], []

        # Gather in lists and encode labels as indices:
        # dict is a dictionary in the form: {'file': <wav file>; 'label': <label>}
        # audio_processor.get_data_from_file() returns the preprocessed data of
        # the audio file in PyTorch tensor and the index of the label.
        # During training (i.e. mode = "training"), background noise will be
        # randomly selected and added to the audio data.
        # During testing (i.e. mode = "testing"), background noise will not be
        # added to the audio data except for silence.
        for dict in batch:
            data, label = audio_processor.get_data_from_file(
                dict, BACKGROUND_FREQUENCY, BACKGROUND_VOLUME_RANGE, \
                TIME_SHIFT_SAMPLE, mode)
            tensors += data
            targets += label

        # Group the list of tensors into a batched tensor
        tensors = pad_sequence(tensors)
        targets = torch.stack(targets)

        return tensors, targets
    return collate_fn_


def make_data_loaders(audio_processor, device,
                      test_batch_size=None, valid_batch_size=None,
                      num_workers = None,
                      return_loaders=['training', 'validation', 'testing']):
    # Create train dataloader
    if device == "cuda":
        if num_workers is None:
            num_workers = 1
        pin_memory = True
    else:
        if num_workers is None:
            num_workers = 0
        pin_memory = False

    train_set = audio_processor.data_index['training']
    test_set = audio_processor.data_index['testing']
    valid_set = audio_processor.data_index['validation']
    print(
        'Train size:', len(train_set),
        "Val size:", len(valid_set),
        "Test size:", len(test_set)
    )
    out = {}
    if 'training' in return_loaders:
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=BATCH_SIZE,
            shuffle=True,
            # During training (i.e. mode = "training"), background noise will be
            # randomly selected and added to the audio data
            collate_fn=make_collate_fn(audio_processor, "training"),
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        out['training'] = train_loader

    # Create test dataloader
    if 'testing' in return_loaders:
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=len(test_set) if test_batch_size is None else test_batch_size,
            shuffle=False,
            drop_last=False,
            # During testing (i.e. mode = "testing"), background noise will not be added
            # to the audio data except for silence
            collate_fn=make_collate_fn(audio_processor, "testing"),
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        out['testing'] = test_loader

    # Create validation loader
    if 'validation' in return_loaders:
        valid_loader = torch.utils.data.DataLoader(
            valid_set,
            batch_size=len(valid_set) if valid_batch_size is None else valid_batch_size,
            shuffle=False,
            drop_last=False,
            # During testing (i.e. mode = "testing"), background noise will not be added
            # to the audio data except for silence
            collate_fn=make_collate_fn(audio_processor, "testing"),
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        out['validation'] = valid_loader

    return out