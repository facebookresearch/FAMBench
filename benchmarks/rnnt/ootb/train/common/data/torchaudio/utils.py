# Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
# Notified per clause 4(b) of the license
from __future__ import print_function

import fnmatch
import io
import os
from tqdm import tqdm
import subprocess
import torch.distributed as dist
import torch


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __len__(self):
        return len(self.word2idx)


class Corpus(object):
    def __init__(self):
        self.dictionary = Dictionary()

    def get_data(self, path, batch_size=20):
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

                    # Tokenize the file content
        ids = torch.LongTensor(tokens)
        token = 0
        with open(path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
        num_batches = ids.size(0) // batch_size
        ids = ids[:num_batches * batch_size]
        return ids.view(batch_size, -1)


def create_manifest(data_path, output_path, min_duration=None, max_duration=None):
    file_paths = [os.path.join(dirpath, f)
                  for dirpath, dirnames, files in os.walk(data_path)
                  for f in fnmatch.filter(files, '*.wav')]
    file_paths = order_and_prune_files(file_paths, min_duration, max_duration)
    with io.FileIO(output_path, "w") as file:
        for wav_path in tqdm(file_paths, total=len(file_paths)):
            transcript_path = wav_path.replace('/wav/', '/txt/').replace('.wav', '.txt')
            sample = os.path.abspath(wav_path) + ',' + os.path.abspath(transcript_path) + '\n'
            file.write(sample.encode('utf-8'))
    print('\n')


def order_and_prune_files(file_paths, min_duration, max_duration):
    print("Sorting manifests...")
    duration_file_paths = [(path, float(subprocess.check_output(
        ['soxi -D \"%s\"' % path.strip()], shell=True))) for path in file_paths]
    if min_duration and max_duration:
        print("Pruning manifests between %d and %d seconds" % (min_duration, max_duration))
        duration_file_paths = [(path, duration) for path, duration in duration_file_paths if
                               min_duration <= duration <= max_duration]

    def func(element):
        return element[1]

    duration_file_paths.sort(key=func)
    return [x[0] for x in duration_file_paths]  # Remove durations

def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= world_size
    return rt
