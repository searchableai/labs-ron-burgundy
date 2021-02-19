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
import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict
import torch.nn.functional as F
from enum import Enum


class TripletsDataset(IterableDataset):
    def __init__(self, model, queries, corpus, triplet_list):
        self.model = model
        self.queries = queries
        self.corpus = corpus
        self.triplet_list = triplet_list

    def __iter__(self):
        for l in self.triplet_list:
            qid, pos_id, neg_id = l
            #print((qid, pos_id, neg_id))
            query_text = self.queries[qid]
            pos_text = self.corpus[pos_id]
            neg_text = self.corpus[neg_id]
            #yield InputExample(texts=[query_text, pos_text, neg_text])
            yield (query_text, pos_text, neg_text)
    def __len__(self):
        return len(self.triplet_list)

class LMDataModule(pl.LightningDataModule):
  def __init__(self, model_name_or_path, training_file, train_batch_size):
    super().__init__()
    self.model_name_or_path = model_name_or_path
    self.training_file = training_file
    self.train_batch_size = train_batch_size
    #self.window_size = window_size
    #self.jump = jump
    self.episode_inputs = []
    self.train_dataset = []

  def prepare_data(self):
    with open(self.training_file, 'r') as read_file:
      training_data = json.load(read_file)
    #trainingd, eval_data = preprocess_data_v2(episode_inputs, self.window_size, self.jump)
    #self.train_queries, self.train_corpus_dict, train_corpus_inv_dict, self.train_triplet_list = prepare_training_triplet(trainingd)
    self.train_queries = training_data['queries']
    self.train_corpus_dict = training_data['corpus_dict']
    self.train_triplet_list = training_data['triplet_list']
    #import pdb; pdb.set_trace()

  def setup(self, stage=None):
    self.train_dataset = TripletsDataset(model=self.model_name_or_path, queries=self.train_queries, corpus=self.train_corpus_dict, triplet_list = self.train_triplet_list)
    #import pdb; pdb.set_trace()

  def train_dataloader(self):
    #import pdb; pdb.set_trace()
    return DataLoader(
        self.train_dataset,
        shuffle=False,
        batch_size=self.train_batch_size
    )
  def val_dataloader(self):
      pass
  def test_dataloader(self):
      pass

class SBModel(pl.LightningModule):
    def __init__(self, model_name, learning_rate, adam_beta1, adam_beta2, adam_epsilon):
        super().__init__()
        self.save_hyperparameters()
        #
        word_embedding_model = models.Transformer(model_name, max_seq_length=512)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        #import pdb; pdb.set_trace()
        td = []
        for i in range(len(batch[0])):
            query_text = batch[0][i]
            pos_text = batch[1][i]
            neg_text = batch[2][i]
           #td.append(InputExample(texts=[query_text, pos_text, neg_text]))
            td.append([query_text, pos_text, neg_text])
        #query_texts = [b[0]
        #for b in batch:
        #    import pdb; pdb.set_trace()
        #    query_text, pos_text, neg_text = b
        #    td.append(InputExample(texts=[query_text, pos_text, neg_text]))
        features = []
        for sample in td:
          t = self.model.tokenize(sample)
          tkeys = t.keys()
          t = {k:t[k].to('cuda') for k in tkeys}
          features.append(t)
        #features = td
        labels = None
        #import pdb; pdb.set_trace()
        train_loss = TripletLoss(model=self.model)
        #import pdb; pdb.set_trace()
        loss_value = train_loss(features, labels)
        self.log('train_loss', loss_value, on_epoch=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(),
                          self.hparams.learning_rate,
                          betas=(self.hparams.adam_beta1,
                                 self.hparams.adam_beta2),
                          eps=self.hparams.adam_epsilon,)
        return optimizer

class MultipleNegativesRankingLoss(nn.Module):
    """
        This loss expects as input a batch consisting of sentence pairs (a_1, p_1), (a_2, p_2)..., (a_n, p_n)
        where we assume that (a_i, p_i) are a positive pair and (a_i, p_j) for i!=j a negative pair.
        For each a_i, it uses all other p_j as negative samples, i.e., for a_i, we have 1 positive example (p_i) and
        n-1 negative examples (p_j). It then minimizes the negative log-likehood for softmax normalized scores.
        This loss function works great to train embeddings for retrieval setups where you have positive pairs (e.g. (query, relevant_doc))
        as it will sample in each batch n-1 negative docs randomly.
        The performance usually increases with increasing batch sizes.
        For more information, see: https://arxiv.org/pdf/1705.00652.pdf
        (Efficient Natural Language Response Suggestion for Smart Reply, Section 4.4)
        You can also provide one or multiple hard negatives per anchor-positive pair by structering the data like this:
        (a_1, p_1, n_1), (a_2, p_2, n_2)
        Here, n_1 is a hard negative for (a_1, p_1). The loss will use for the pair (a_i, p_i) all p_j (j!=i) and all n_j as negatives.
        Example::
            from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler, losses
            from sentence_transformers.readers import InputExample
            model = SentenceTransformer('distilbert-base-nli-mean-tokens')
            train_examples = [InputExample(texts=['Anchor 1', 'Positive 1']),
                InputExample(texts=['Anchor 2', 'Positive 2'])]
            train_dataset = SentencesDataset(train_examples, model)
            train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
            train_loss = losses.MultipleNegativesRankingLoss(model=model)
    """
    def __init__(self, model: SentenceTransformer, scale: float = 20.0, similarity_fct = util.pytorch_cos_sim):
        """
        :param model: SentenceTransformer model
        :param scale: Output of similarity function is multiplied by scale value
        :param similarity_fct: similarity function between sentence embeddings. By default, cos_sim. Can also be set to dot product (and then set sclae to 1)
        """
        super(MultipleNegativesRankingLoss, self).__init__()
        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()


    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        reps_a = reps[0]
        reps_b = torch.cat(reps[1:])
        return self.multiple_negatives_ranking_loss(reps_a, reps_b)


    def multiple_negatives_ranking_loss(self, embeddings_a: Tensor, embeddings_b: Tensor):
        """
        :param embeddings_a:
            Tensor of shape (batch_size, embedding_dim)
        :param embeddings_b:
            Tensor of shape (batch_size, embedding_dim)
        :return:
            The scalar loss
        """
        scores = self.similarity_fct(embeddings_a, embeddings_b) * self.scale
        labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)  # Example a[i] should match with b[i]
        return self.cross_entropy_loss(scores, labels)


class TripletDistanceMetric(Enum):
    """
    The metric for the triplet loss
    """
    COSINE = lambda x, y: 1 - F.cosine_similarity(x, y)
    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)


class TripletLoss(nn.Module):
    """
    This class implements triplet loss. Given a triplet of (anchor, positive, negative),
    the loss minimizes the distance between anchor and positive while it maximizes the distance
    between anchor and negative. It compute the following loss function:
    loss = max(||anchor - positive|| - ||anchor - negative|| + margin, 0).
    Margin is an important hyperparameter and needs to be tuned respectively.
    For further details, see: https://en.wikipedia.org/wiki/Triplet_loss
    :param model: SentenceTransformerModel
    :param distance_metric: Function to compute distance between two embeddings. The class TripletDistanceMetric contains common distance metrices that can be used.
    :param triplet_margin: The negative should be at least this much further away from the anchor than the positive.
    Example::
        from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler, losses
        from sentence_transformers.readers import InputExample
        model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        train_examples = [InputExample(texts=['Anchor 1', 'Positive 1', 'Negative 1']),
            InputExample(texts=['Anchor 2', 'Positive 2', 'Negative 2'])]
        train_dataset = SentencesDataset(train_examples, model)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
        train_loss = losses.TripletLoss(model=model)
    """
    def __init__(self, model: SentenceTransformer, distance_metric=TripletDistanceMetric.EUCLIDEAN, triplet_margin: float = 5):
        super(TripletLoss, self).__init__()
        self.model = model
        self.distance_metric = distance_metric
        self.triplet_margin = triplet_margin


    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        #print(sentence_features)

        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]

        r_anchor_tuple = tuple(r[0].unsqueeze(dim = 1) for r in reps)
        rep_anchor = torch.cat(r_anchor_tuple, dim = 1)
        rep_anchor = torch.transpose(rep_anchor, 0, 1)

        r_pos_tuple = tuple(r[1].unsqueeze(dim = 1) for r in reps)
        rep_pos = torch.cat(r_pos_tuple, dim = 1)
        rep_pos = torch.transpose(rep_pos, 0, 1)

        r_neg_tuple = tuple(r[2].unsqueeze(dim = 1) for r in reps)
        rep_neg = torch.cat(r_neg_tuple, dim = 1)
        rep_neg = torch.transpose(rep_neg, 0, 1)

        distance_pos = self.distance_metric(rep_anchor, rep_pos)
        distance_neg = self.distance_metric(rep_anchor, rep_neg)

        losses = F.relu(distance_pos - distance_neg + self.triplet_margin)
        return losses.mean()

if __name__ == '__main__':
    data_file = "./data/training_data_triplet_ws2-j1.json"
    train_batch_size = 32
    #window_size = 2
    #jump = 1
    model_name = 'roberta-base'
    learning_rate = 0.001
    adam_beta1 = 0.9
    adam_beta2 = 0.99
    adam_epsilon = 1e-8

    data_module = LMDataModule(
        model_name_or_path=model_name,
        training_file=data_file,
        train_batch_size=train_batch_size,
        #window_size = window_size,
        #jump = jump
    )

    sbmodel = SBModel(
        model_name=model_name,
        learning_rate=learning_rate,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_epsilon=adam_epsilon,
    )

    # Initialize a trainer
    trainer = pl.Trainer(gpus=1, max_epochs=3, progress_bar_refresh_rate=20)

    # Train the model ⚡
    trainer.fit(sbmodel, data_module)
