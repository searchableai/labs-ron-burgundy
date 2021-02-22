from argparse import ArgumentParser
import pytorch_lightning as pl
from transformers.optimization import AdamW
#from data import LMDataModule
import itertools
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, LoggingHandler, util, models, evaluation, losses, InputExample
import logging
from datetime import datetime
import gzip
import os
import tarfile
from collections import defaultdict
from torch.utils.data import IterableDataset
import random
import json
from tqdm import tqdm
import os
from pathlib import Path
def preprocess_data_v2(episode_inputs, window_size, jump = 1):
  trainingd = []
  eval_data = []

  for episode_id, episode_info in episode_inputs.items():

    turns = episode_info['turns']
    context = episode_info['context']

    # define max_turn to ensure that the next_turn does not exceed the list.
    max_turn = max([int(l) for l in turns.keys()])

    for turn, turn_info in turns.items():
      # turn_info contains gold_sent_indices and dialogue history.
      next_turn = str(int(turn) + jump)
      # avoid two cases:
      # if jump is zero -> the first turn doesn't have history
      # next turn shouldn't exceed the max_turns.
      if int(next_turn) <= max_turn and int(next_turn) > 0:
        # split the data for eval/training
        # for the eval, keep the history and the kg_sents for the next turn
        # for the training, form the dataset
        r = random.randint(1,100)

        if r <= 5:
          mode = "eval"
        else:
          mode = "training"

        # Only consider the cases with at least one gold and one negative (0).
        # The relevant sentences of next turn
        gold_ind = turns[next_turn]['gold_sent_indices']
        gold_count = sum([1 if l == 1 else 0 for l in gold_ind])
        negative_count = sum([1 if l == 0 else 0 for l in gold_ind])
        all_count = min(gold_count, negative_count)

        # # Define the windowed dialogue history
        history_dict = turns[next_turn]['dialogue_history']
        ordered_turns = sorted([int(l) for l in list(history_dict.keys())])
        len_ = len(ordered_turns)
        window = min(len_, window_size)
        selected_keys = [str(l) for l in ordered_turns[len_ - window:]]
        history_dict = {k:history_dict[k] for k in selected_keys}
        history = ' '. join([history_dict[k] for k in selected_keys])

        if all_count >= 1:
          if mode == "eval":
            l = {}
            l['queries'] = history
            l['corpus'] = context
            l['next_gold_ind'] = gold_ind
            eval_data.append(l)

          elif mode == "training":
            l = {}
            l['queries'] = history
            l['corpus'] = context
            l['next_gold_ind'] = gold_ind
            trainingd.append(l)
            '''
            gold_sents = [context[i] for i in range(len(context)) if gold_ind[i] == 1]
            if all_count != gold_count:
              gold_sents = random.sample(gold_sents, all_count)

            # randomly select a subset of the negative samples
            neg_sents = [context[i] for i in range(len(context)) if gold_ind[i] == 0]
            if all_count != negative_count:
              neg_sents = random.sample(neg_sents, all_count)

            #ccc = 0
            #if len(history) == 0:
            #    ccc += 1
            #    print(ccc)

            # append the data
            for gsent in gold_sents:
              trainingd.append([history, gsent, 1])
            for nsent in neg_sents:
              trainingd.append([history, nsent, 0])

            '''
  #assert sum([1 for t in trainingd if len(t[0]) == 0]) == 0, 'There are empty history strings.'
  return trainingd, eval_data

def prepare_training_triplet(trainingd):
  '''
  The outputs of this function are:
  1) queries (qid: query)
  2) corpus (cid: corpus)
  3) triplet list
  '''
  balanced = True
  queries = {}
  corpus_dict = {} # cid: corpus
  corpus_inv_dict = {} # corpus: cid
  triplet_list = []
  qid = 0
  cid = 0

  for sample in tqdm(trainingd):
    # prepare the queries (histories)
    history = sample['queries']
    queries[str(qid)] = history


    # prepare the queries
    # ensure no duplicates
    corpus = sample['corpus']
    for c in corpus:
      if c not in corpus_inv_dict:
        corpus_dict[str(cid)] = c
        corpus_inv_dict[c] = str(cid)
        cid += 1

    # prepare the triplet list
    gold_ind = sample['next_gold_ind']
    pos_samples = [corpus[i] for i in range(len(corpus)) if gold_ind[i] == 1]
    neg_samples = [corpus[i] for i in range(len(corpus)) if gold_ind[i] == 0]

    # check the balanced format
    # both true and false resulted in the same length of triplet list.
    if balanced:
      gold_count = sum([1 if l == 1 else 0 for l in gold_ind])
      negative_count = sum([1 if l == 0 else 0 for l in gold_ind])
      all_count = min(gold_count, negative_count)
      if all_count != gold_count:
        pos_samples = random.sample(pos_samples, all_count)
      if all_count != negative_count:
        neg_samples = random.sample(neg_samples, all_count)

    # find the cid for pos samples
    pos_ids = [corpus_inv_dict[p] for p in pos_samples]
    neg_ids = [corpus_inv_dict[n] for n in neg_samples]
    pos_neg_list = [list(l) for l in itertools.chain(itertools.product(pos_ids, neg_ids))]

    # append to triplet list
    for pn in pos_neg_list:
      triplet_list.append([str(qid)] + pn)
      #if qid < 10:
        #print(triplet_list)
    qid += 1
  return queries, corpus_dict, corpus_inv_dict, triplet_list

if __name__ == '__main__':
    random.seed(42)
    with open("./data/all_data_v2.json", 'r') as read_file:
        episode_inputs = json.load(read_file)
    window_size = 10
    jump = 1
    data_save_path = './data/'
    training_data_save_name = data_save_path + f'training_data_triplet_ws{window_size}_j{jump}.json'
    eval_data_save_name = data_save_path + f'eval_data_ws{window_size}_j{jump}.json'
    trainingd, eval_data = preprocess_data_v2(episode_inputs, window_size = window_size, jump = jump)
    queries, corpus_dict, corpus_inv_dict, triplet_list = prepare_training_triplet(trainingd)
    training_data = {}
    training_data['queries'] = queries
    training_data['corpus_dict'] = corpus_dict
    training_data['triplet_list'] = triplet_list
    Path(data_save_path).mkdir(parents=True, exist_ok=True)
    with open(training_data_save_name, 'w') as write_file:
        json.dump(training_data, write_file, indent = 2)
    with open(eval_data_save_name, 'w') as write_file:
        json.dump(eval_data, write_file, indent = 2)
