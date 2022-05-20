import os
import subprocess
from tempfile import NamedTemporaryFile

from torch.distributed import get_rank
from torch.distributed import get_world_size
# from torch.distributed.deprecated import get_rank
# from torch.distributed.deprecated import get_world_size
from torch.utils.data.sampler import Sampler

import random
import numpy as np
import scipy.signal
import torch
import torch.functional as F
import torchaudio
import math
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from common.data.dataset import AudioDataset

#TODO This load fails, but we don't need it anyways, only a small part of this file is used anymore(AudioDataLoader).  We should move it.
#from data.SpecAugment import sparse_image_warp_zcaceres

windows = {'hamming': scipy.signal.hamming, 'hann': scipy.signal.hann, 'blackman': scipy.signal.blackman,
           'bartlett': scipy.signal.bartlett}

def load_audio(path):
    sound, _ = torchaudio.load(path)
    sound = sound.numpy().T
    if len(sound.shape) > 1:
        if sound.shape[1] == 1:
            sound = sound.squeeze()
        else:
            sound = sound.mean(axis=1)  # multiple channels, average
    return sound


class AudioParser(object):
    def parse_transcript(self, transcript_path):
        """
        :param transcript_path: Path where transcript is stored from the manifest file
        :return: Transcript in training/testing format
        """
        raise NotImplementedError

    def parse_audio(self, audio_path):
        """
        :param audio_path: Path where audio is stored from the manifest file
        :return: Audio in training/testing format
        """
        raise NotImplementedError


def _collate_fn(batch):
    bs = len(batch)
    max_len = lambda l, idx: max(el[idx].size(0) for el in l)
    max_len2 = lambda l, idx: max(el[0][idx].size(0) for el in l)
    #TODO, don't use 80...maybe get shape of 2nd dimension of batch
    device = "cuda:"+str(batch[0][0].get_device()) if batch[0][0].get_device() >= 0 else "cpu"

    audio = torch.zeros(bs, 80, max_len2(batch, 0)).to(device)
    audio_lens = torch.zeros(bs, dtype=torch.int32)
    transcript = torch.zeros(bs, max_len(batch, 2)).to(device)
    transcript_lens = torch.zeros(bs, dtype=torch.int32)




    for i, sample in enumerate(batch):
        audio[i].narrow(1, 0, sample[0].size(1)).copy_(sample[0])
        audio_lens[i] = sample[1]
        transcript[i].narrow(0, 0, sample[2].size(0)).copy_(sample[2])
        transcript_lens[i] = sample[3]

    return audio, audio_lens, transcript, transcript_lens


class AudioDataLoader(DataLoader):
    def __init__(self,
                 #config
                 config_features,
                 pipeline_type,
                 gpu_id,
                 # DataSet
                 data_dir, manifest_fpaths,
                 tokenizer,
                 sample_rate=16000, min_duration=0.1, max_duration=float("inf"),
                 max_utts=0, normalize_transcripts=True,
                 trim_silence=False,
                 speed_perturbation=None,
                 ignore_offline_speed_perturbation=False,
                 # Sampler
                 batch_size=1, num_replicas=None, rank=None,
                 # Loader
                 shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None, *, prefetch_factor=2,
                 persistent_workers=False,
                 device_type="gpu"):
        """
        Creates a data loader for AudioDatasets.
        """
        self.audio_dataset = AudioDataset(
                 data_dir=data_dir,
                 tokenizer=tokenizer,
                 manifest_fpaths=manifest_fpaths,
                 n_filt=config_features["n_filt"],
                 n_fft=config_features["n_fft"],
                 device_type=device_type,
                 gpu_id=gpu_id
                 )

        if batch_sampler is None:
            if num_replicas <= 1:
                self.batch_sampler = BucketingSampler(self.audio_dataset,
                                           batch_size=batch_size)
            else:
                self.batch_sampler = DistributedBucketingSampler(self.audio_dataset,
                                            batch_size=batch_size,
                                            num_replicas=num_replicas,
                                            rank=rank)
        else:
            self.batch_sampler = batch_sampler

        self.pipeline_type = pipeline_type
        self.gpu_id = gpu_id
        self.device_type = device_type

        super(AudioDataLoader, self).__init__(self.audio_dataset,
                                            batch_sampler=self.batch_sampler,
                                            num_workers=num_workers, collate_fn=None,
                                            pin_memory=pin_memory,
                                            timeout=timeout, worker_init_fn=worker_init_fn,
                                            prefetch_factor=prefetch_factor,
                                            persistent_workers=persistent_workers)
        self.collate_fn = _collate_fn


class BucketingSampler(Sampler):
    def __init__(self, data_source, batch_size=1):
        """
        Samples batches assuming they are in order of size to batch similarly sized samples together.
        """
        super(BucketingSampler, self).__init__(data_source)
        self.data_source = data_source
        ids = list(range(0, len(data_source)))
        self.bins = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]

    def __iter__(self):
        for ids in self.bins:
            #numpy is going to pull work back to cpu
            #np.random.shuffle(ids)
            yield ids

    def __len__(self):
        return len(self.bins)

    def shuffle(self, epoch):
        np.random.shuffle(self.bins)


class BatchRandomSampler(Sampler):
    """
    Batches the data consecutively and randomly samples
    by batch without replacement.
    """

    def __init__(self, data_source, batch_size):
        it_end = len(data_source) - batch_size + 1
        self.batches = [range(i, i + batch_size)
                for i in range(0, it_end, batch_size)]
        self.data_source = data_source

    def __iter__(self):
        random.shuffle(self.batches)
        return (i for b in self.batches for i in b)

    def __len__(self):
        return len(self.data_source)


class DistributedBucketingSampler(Sampler):
    def __init__(self, data_source, batch_size=1, num_replicas=None, rank=None):
        """
        Samples batches assuming they are in order of size to batch similarly sized samples together.
        """
        super(DistributedBucketingSampler, self).__init__(data_source)
        if num_replicas is None:
            num_replicas = get_world_size()
        if rank is None:
            rank = get_rank()
        self.data_source = data_source
        self.ids = list(range(0, len(data_source)))
        self.batch_size = batch_size
        self.bins = [self.ids[i:i + batch_size] for i in range(0, len(self.ids), batch_size)]
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.bins) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        offset = self.rank
        # add extra samples to make it evenly divisible
        bins = self.bins + self.bins[:(self.total_size - len(self.bins))]
        assert len(bins) == self.total_size
        samples = bins[offset::self.num_replicas]  # Get every Nth bin, starting from rank
        return iter(samples)

    def __len__(self):
        return self.num_samples

    def shuffle(self, epoch):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(epoch)
        bin_ids = list(torch.randperm(len(self.bins), generator=g))
        self.bins = [self.bins[i] for i in bin_ids]
