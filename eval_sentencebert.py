import json
import random
from sentence_transformers import SentencesDataset, SentenceTransformer, evaluation, InputExample
from sentence_transformers.evaluation import SentenceEvaluator
from torch.utils.data import DataLoader
from sentence_transformers import losses, util
from typing import List, Tuple, Dict, Set
from tqdm import tqdm
import logging
from torch import Tensor, device
import torch
import numpy as np
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

def pytorch_cos_sim(a: Tensor, b: Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    This function can be used as a faster replacement for 1-scipy.spatial.distance.cdist(a,b)
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))

def preprocess_data(episode_inputs, window_size, jump = 1):
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

        if r <= 10:
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
  assert sum([1 for t in trainingd if len(t[0]) == 0]) == 0, 'There are empty history strings.'
  return trainingd, eval_data

def prepare_ir_eval_data_2(eval_data):
    # Prepare the eval data for InformationRetrieval evaluation

    queries = {}
    related_corpus_per_query = {}
    gold_sents_idx = {}
    qid = 0

    for ed in eval_data:

      # prepare the queries (histories)
      queries[str(qid)] = ed['queries']

      # create a corpus set for evaluation of each query
      corpus_list = ed['corpus']
      related_corpus_per_query[str(qid)] = {str(idx): v for idx, v in enumerate(corpus_list)}

      # the idx of the gold sentences
      gold_ind = ed['next_gold_ind']

      #assert isinstance(related_corpus_per_query[qid], list), "corpus for each query should be a list"
      gold_sents_idx[str(qid)] = set([str(idx) for idx, v in enumerate(gold_ind) if v == 1])

      qid += 1

    return queries, related_corpus_per_query, gold_sents_idx

class InformationRetrievalEvaluator(SentenceEvaluator):
    def __init__(self,
                     queries: Dict[str, str],  #qid => query
                     corpus: Dict[str, str],  #qid => dict[cid: doc]
                     relevant_docs: Dict[str, Set[str]],  #qid => Set[cid]
                     mrr_at_k: List[int] = [10],
                     ndcg_at_k: List[int] = [10],
                     accuracy_at_k: List[int] = [1, 3, 5, 10],
                     precision_recall_at_k: List[int] = [1, 3, 5, 10],
                     map_at_k: List[int] = [100],
                     show_progress_bar: bool = False,
                     batch_size = 32,
      ):

          self.queries_ids = []
          for qid in queries:
              if qid in relevant_docs and len(relevant_docs[qid]) > 0:
                  self.queries_ids.append(qid)
          self.queries = [queries[qid] for qid in self.queries_ids]
          self.corpus = [corpus[qid] for qid in self.queries_ids]
          self.relevant_docs = relevant_docs
          self.mrr_at_k = mrr_at_k
          self.ndcg_at_k = ndcg_at_k
          self.accuracy_at_k = accuracy_at_k
          self.precision_recall_at_k = precision_recall_at_k
          self.map_at_k = map_at_k
          self.batch_size = batch_size

          self.show_progress_bar = show_progress_bar

    def __call__(self, model, output_path: str = None):

        max_k = max(max(self.mrr_at_k), max(self.ndcg_at_k), max(self.accuracy_at_k), max(self.precision_recall_at_k), max(self.map_at_k))
        query_embeddings = model.encode(self.queries, show_progress_bar=self.show_progress_bar, batch_size=self.batch_size, convert_to_tensor=True)
        #import pdb; pdb.set_trace()
        queries_result_list = [[] for _ in range(len(query_embeddings))]
        
        # for one query:
        for qidx in tqdm(range(len(query_embeddings))):
          curpus_ids = []
          curCorp_dict = {}
          curCorp = []
          curQEmbedding = query_embeddings[qidx]
          #import pdb; pdb.set_trace()
          #queries_result_list = [[] for _ in range(len(query_embeddings))]
          #print(qidx)
          curCorp_dict = self.corpus[qidx]
          corpus_ids = list(curCorp_dict.keys())
          #import pdb; pdb.set_trace()
          curCorp = [curCorp_dict[cid] for cid in corpus_ids]
          #import pdb; pdb.set_trace()
          corpus_embeddings = model.encode(curCorp, show_progress_bar=False, batch_size=self.batch_size, convert_to_tensor=True)
          #import pdb; pdb.set_trace()
          cos_scores = pytorch_cos_sim(curQEmbedding, corpus_embeddings)
          #import pdb; pdb.set_trace()
          del corpus_embeddings

          #Get top-k values
          # the cos_scores should be 1 * the number of corpus articles for the following.
          cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, min(max_k, len(cos_scores[0])), dim=1, largest=True, sorted=False)
          #import pdb; pdb.set_trace()
          cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
          cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()
          del cos_scores

          for corpus_id, score in zip(cos_scores_top_k_idx[0], cos_scores_top_k_values[0]):
            cid = corpus_ids[corpus_id]
            queries_result_list[qidx].append({'corpus_id': cid, 'score': score})

        scores = self.compute_metrics(queries_result_list)
        return scores, queries_result_list

    def compute_metrics(self, queries_result_list: List[object]):
        # Init score computation values
        num_hits_at_k = {k: 0 for k in self.accuracy_at_k}
        precisions_at_k = {k: [] for k in self.precision_recall_at_k}
        recall_at_k = {k: [] for k in self.precision_recall_at_k}
        MRR = {k: 0 for k in self.mrr_at_k}
        ndcg = {k: [] for k in self.ndcg_at_k}
        AveP_at_k = {k: [] for k in self.map_at_k}

        # Compute scores on results
        for query_itr in range(len(queries_result_list)):
            query_id = self.queries_ids[query_itr]

            # Sort scores
            top_hits = sorted(queries_result_list[query_itr], key=lambda x: x['score'], reverse=True)
            query_relevant_docs = self.relevant_docs[query_id]

            # Accuracy@k - We count the result correct, if at least one relevant doc is accross the top-k documents
            for k_val in self.accuracy_at_k:
                for hit in top_hits[0:k_val]:
                    if hit['corpus_id'] in query_relevant_docs:
                        num_hits_at_k[k_val] += 1
                        break

            # Precision and Recall@k
            for k_val in self.precision_recall_at_k:
                num_correct = 0
                for hit in top_hits[0:k_val]:
                    if hit['corpus_id'] in query_relevant_docs:
                        num_correct += 1

                precisions_at_k[k_val].append(num_correct / k_val)
                recall_at_k[k_val].append(num_correct / len(query_relevant_docs))

            # MRR@k
            for k_val in self.mrr_at_k:
                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit['corpus_id'] in query_relevant_docs:
                        MRR[k_val] += 1.0 / (rank + 1)
                        break

            # NDCG@k
            for k_val in self.ndcg_at_k:
                predicted_relevance = [1 if top_hit['corpus_id'] in query_relevant_docs else 0 for top_hit in top_hits[0:k_val]]
                true_relevances = [1] * len(query_relevant_docs)

                ndcg_value = self.compute_dcg_at_k(predicted_relevance, k_val) / self.compute_dcg_at_k(true_relevances, k_val)
                ndcg[k_val].append(ndcg_value)

            # MAP@k
            for k_val in self.map_at_k:
                num_correct = 0
                sum_precisions = 0

                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit['corpus_id'] in query_relevant_docs:
                        num_correct += 1
                        sum_precisions += num_correct / (rank + 1)

                avg_precision = sum_precisions / min(k_val, len(query_relevant_docs))
                AveP_at_k[k_val].append(avg_precision)

          # Compute averages
        for k in num_hits_at_k:
            num_hits_at_k[k] /= len(self.queries)

        for k in precisions_at_k:
            precisions_at_k[k] = np.mean(precisions_at_k[k])

        for k in recall_at_k:
            recall_at_k[k] = np.mean(recall_at_k[k])

        for k in ndcg:
            ndcg[k] = np.mean(ndcg[k])

        for k in MRR:
            MRR[k] /= len(self.queries)

        for k in AveP_at_k:
            AveP_at_k[k] = np.mean(AveP_at_k[k])


        return {'accuracy@k': num_hits_at_k, 'precision@k': precisions_at_k, 'recall@k': recall_at_k, 'ndcg@k': ndcg, 'mrr@k': MRR, 'map@k': AveP_at_k}

    @staticmethod
    def compute_dcg_at_k(relevances, k):
        dcg = 0
        for i in range(min(len(relevances), k)):
            dcg += relevances[i] / np.log2(i + 2)  #+2 as we start our idx at 0
        return dcg

if __name__ == '__main__':

    with open("../all_data.json", 'r') as read_file:
        episode_inputs = json.load(read_file)
    random.seed(42)
    trainingd, eval_data = preprocess_data(episode_inputs, window_size = 2, jump = 1)
    _, eval_data = train_test_split(eval_data, test_size=0.1, random_state=42)
    model = SentenceTransformer('./output/texp_6_roberta_base_evalclass_64/')
    queries, related_corpus_per_query, relevant_docs = prepare_ir_eval_data_2(eval_data)
    query_ids = list(queries.keys())
    l = InformationRetrievalEvaluator(queries, related_corpus_per_query, relevant_docs, show_progress_bar = True)
    scores, query_scores= l(model)
    all_results = {}
    all_results['scores'] = scores
    all_results['query_scores'] = query_scores
    with open('eval_results.json', 'w') as write_file:
        json.dump(all_results, write_file, indent = 2)
