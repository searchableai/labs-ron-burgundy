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
import pathlib

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

def prepare_ir_eval_data_2(eval_data):
    '''
    Prepare the evaluation data for InformationRetrieval evaluation.
    Inputs:
        eval_data
            a list containing the samples. each element of the list is a dictionary
            including 'queries', 'corpus', and 'next_gold_ind'.

    Output:
        Three dictionaries:
            - queries where the keys are the query ids and the values are the queries.
            - related_corpus_per_query where the keys are the query ids and the values are
              a dictionary including the corpus related to this query.
            - gold_sent_idx where the keys are the query ids and the values are sets
              including the golden corpus ids.
    '''
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
          self.corpus_dict = corpus
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

        # Find the query embeddings.
        # A list with the length equal to the number of queries.
        # Each element of the list is a tensor of shape [768] (roberta-base).
        query_embeddings = model.encode(self.queries, show_progress_bar=self.show_progress_bar, batch_size=self.batch_size, convert_to_tensor=True)

        queries_result_list = [[] for _ in range(len(query_embeddings))]
        all_text = []
        text_output = True
        # for one query:
        for qidx in tqdm(range(len(query_embeddings))):
          curpus_ids = []
          curCorp_dict = {}
          curCorp = []
          curQEmbedding = query_embeddings[qidx]
          curCorp_dict = self.corpus[qidx]
          corpus_ids = list(curCorp_dict.keys())
          curCorp = [curCorp_dict[cid] for cid in corpus_ids]
          corpus_embeddings = model.encode(curCorp, show_progress_bar=False, batch_size=self.batch_size, convert_to_tensor=True)
          # The cos_scores has the shape: torch.Size([1, len(corpus_embeddings)])
          cos_scores = pytorch_cos_sim(curQEmbedding, corpus_embeddings)
          del corpus_embeddings

          # values and indices. both torch.Size([1, 100])
          cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, min(max_k, len(cos_scores[0])), dim=1, largest=True, sorted=False)
          cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
          cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()
          if text_output:
              all_text.append(self.raw_text_output(cos_scores, curCorp, qidx))  
          del cos_scores

          for corpus_id, score in zip(cos_scores_top_k_idx[0], cos_scores_top_k_values[0]):
            cid = corpus_ids[corpus_id]
            queries_result_list[qidx].append({'corpus_id': cid, 'score': score})

        scores = self.compute_metrics(queries_result_list)
        return scores, queries_result_list, all_text

    def raw_text_output(self, cos_scores, curCorp, qidx):
        top_k_values, top_k_idx = torch.topk(cos_scores, min(5, len(cos_scores[0])), dim=1, largest=True, sorted=True)
        if len(cos_scores[0]) < 5: print(len(cos_scores[0]))
        text_output = {}
        retrieved_sample = []
        
        top_k_values = top_k_values.cpu().tolist()
        top_k_idx = top_k_idx.cpu().tolist()
        for corpus_id, score in zip(top_k_idx[0], top_k_values[0]):
            retrieved_sample.append([score, curCorp[corpus_id]])
        text_output['retrieved'] = retrieved_sample
        text_output['query'] = self.queries[qidx]
        rel = self.relevant_docs[self.queries_ids[qidx]]
        c = self.corpus_dict[self.queries_ids[qidx]]
        text_output['ground_truth'] = [c[l] for l in rel]
        #import pdb;pdb.set_trace()
        return text_output

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
    
    from pl_implementation_3 import SBModel
    data_path = './data'
    window = 10
    jump = 1
    eval_data_name = data_path + f'/eval_data_ws{window}_j{jump}.json'

    with open(eval_data_name, 'r') as read_file:
        eval_data = json.load(read_file)

    # In case we want to evaluate a smaller subset, uncomment the following line.
    _, eval_data = train_test_split(eval_data, test_size=0.03, random_state=42)
    

    # The following lines can be used to load a sentence transformer model.
    #model_path ='./output/texp_9_roberta_base_evalclass_64_margin1_256_window5/' 
    # The following line should be linked to the pl checkpoint.
    #model = SentenceTransformer(model_path)

    model_path = "./lightning_logs/version_1/checkpoints/epoch=0-step=8786.ckpt"
    sbmodel = SBModel.load_from_checkpoint(model_path)
    #import pdb; pdb.set_trace()
    model = sbmodel.model


    queries, related_corpus_per_query, relevant_docs = prepare_ir_eval_data_2(eval_data)
    query_ids = list(queries.keys())
    l = InformationRetrievalEvaluator(queries, related_corpus_per_query, relevant_docs, show_progress_bar = True)
    scores, query_scores, text_results = l(model)
    all_results = {}
    print(f'scores : {scores}')
    all_results['scores'] = scores
    all_results['query_scores'] = query_scores
    output_path = model_path + 'eval/'
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    eval_name = output_path + f'eval_results_w{window}_j{1}.json'
    eval_name_text = output_path + f'text_eval_results_w{window}_j{1}.json'
    with open(eval_name, 'w') as write_file:
        json.dump(all_results, write_file, indent = 2)

    with open(eval_name_text, 'w') as write_file:
        json.dump(text_results, write_file, indent = 2)

