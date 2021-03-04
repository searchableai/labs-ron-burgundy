"""
This Memformer Dataloader ignores sentence boundary

It is hard to debug inside the dataloader and processor.
Therefore, please make sure everything works before feeding into FlyDataloader
"""
import os
import json
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset
from torch.nn.utils.rnn import pad_sequence

from tokenizers import ByteLevelBPETokenizer
from nlp import Dataset

from torchfly.rl.env import Env
from torchfly.flydata import FlyDataLoader
from torchfly.flyconfig import GlobalFlyConfig
from torchfly.rl.vector import AsyncVectorEnv
from torchfly.common import set_random_seed, get_rank

from typing import Iterator, Tuple, List

from .text_tasks import MultiTextTask, LanguageModel

# pylint:disable=no-member


class CollateFunc:
    def __init__(self, config):
        self.memory_reset_dropout = config.processing.memory_reset_dropout
        self.time_horizon = config.processing.time_horizon
        self.pad_token_id = config.processing.pad_token_id

    def collate_func(self, observations: List, infos: List, dones: List):
        rollout = []
        memory_reset_signals = []

        for time in range(self.time_horizon):

            source = [obs[time][0] for obs in observations]
            target = [obs[time][1] for obs in observations]
            reset = torch.FloatTensor([info[time] for info in infos])

            input_ids = pad_sequence(
                [torch.LongTensor(item) for item in source], batch_first=True, padding_value=self.pad_token_id
            )

            target_ids = pad_sequence(
                [torch.LongTensor(item) for item in target], batch_first=True, padding_value=self.pad_token_id
            )

            batch = {"decoder_target_ids": target_ids, "decoder_input_ids": input_ids}
            rollout.append(batch)
            memory_reset_signals.append(reset)

        return rollout, memory_reset_signals


class FixedDataLoader:
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size

    def __iter__(self):
        for item in self.data:
            yield item

    def __len__(self):
        return len(self.data)


class TextDataLoaderHelper:
    def __init__(self, config):
        self.config = config
        self.collator = CollateFunc(config.flydata.training)
        # self.eval_collator = CollateFunc(config.flydata.validation)

    def train_dataloader_fn(self, config):
        def _make_env(rank):
            env = TimeTextDataLoader(rank, config.flydata.training)
            return env

        in_series = config.flydata.training.dataloader.in_series
        vec_env = AsyncVectorEnv(
            [_make_env for i in range(config.flydata.training.dataloader.batch_size)], in_series=in_series
        )
        dataloader = FlyDataLoader(config.flydata.training, vec_env, collate_func=self.collator.collate_func)

        return dataloader

    def valid_dataloader_fn(self, config):
        data = torch.load(config.flydata.validation.datapath)
        batch_size = config.flydata.validation.batch_size
        dataloader = FixedDataLoader(data, batch_size)
        return dataloader

    def test_dataloader_fn(self, config):
        data = torch.load(config.flydata.test.datapath)
        batch_size = config.flydata.test.batch_size
        dataloader = FixedDataLoader(data, batch_size)
        return dataloader


class TextDataLoader(Env):
    def __init__(self, rank, config):
        super().__init__()
        self.rank = rank
        self.config = config
        self.filename = config.datapath
        self.batch_size = config.dataloader.batch_size
        self.tokenizer = ByteLevelBPETokenizer(vocab=config.tokenizer.vocab_file, merges=config.tokenizer.merges_file)

        # Load data and select ranking
        dataset = Dataset.from_file(self.filename)
        split_size = max(1, len(dataset) // self.batch_size)
        self.data = dataset[self.rank * split_size:(self.rank + 1) * split_size]["document"]

        self.document_processor = LanguageModel(config, self.tokenizer)

        # random.seed(rank)
        self.iterator = iter(self)

    def step(self, actions=None):
        item = next(self.iterator)
        source, target, is_last_segment, done = item
        observation = (source, target)
        info = is_last_segment
        return observation, info, done

    def reset(self):
        self.document_processor.reset()
        self.iterator = iter(self)

    def __iter__(self):
        """ This iterator does not end
        Returns:
            done: bool, when `self.data` are all sampled
        """
        random.shuffle(self.data)

        for document in self.data:
            for item in self.document_processor.process_document(document):
                source, target, is_last_segment = item
                done = False
                yield source, target, is_last_segment, done

        # if all documents have been processed
        # we continue and wait until the AsyncEnv
        while True:
            done = True
            yield [], [], 0, done


class TimeTextDataLoader(TextDataLoader):
    def __init__(self, rank, config):
        super().__init__(rank, config)
        self.time_horizon = config.processing.time_horizon

    def step(self, actions=None):
        observations = []
        infos = []

        for _ in range(self.time_horizon):
            item = next(self.iterator)
            source, target, is_last_segment, done = item
            obs = (source, target)
            info = is_last_segment

            observations.append(obs)
            infos.append(info)

        done = done

        return observations, infos, done
