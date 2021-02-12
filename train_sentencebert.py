import json
import random
from sentence_transformers import SentencesDataset, SentenceTransformer, evaluation, InputExample
from torch.utils.data import DataLoader
from sentence_transformers import losses, util


def prepare_ir_eval_data(eval_data):
  # Prepare the eval data for InformationRetrieval evaluation

  queries = {}
  corpus = {}
  inv_corpus = {}
  next_turn_relevant_sents = {}
  qid = 0
  cid = 0
  seen = set()

  for ed in eval_data:

    qid_cid = set()

    # prepare the corpus
    # add the gold sentences.
    # gold sents for one turn will be distraction for another.
    gold_ind = ed['next_gold_ind']
    context = ed['corpus']

    gold_sents = [context[i] for i in range(len(context)) if gold_ind[i] == 1]
    #print(len(gold_sents))
    #neg_sents = [context[i] for i in range(len(context)) if gold_ind[i] == 0]
    #all_sents = gold_sents + neg_sents

    # prepare the queries (histories)
    queries[qid] = ed['queries']

    # ensure no duplicates
    for sent in gold_sents:
      if sent not in seen:
        seen.add(sent)
        corpus[cid] = sent
        inv_corpus[sent] = cid
        qid_cid.add(cid)
        cid += 1
      else:
        qid_cid.add(inv_corpus[sent])
    assert(len(gold_sents) == len(qid_cid)), "not the same length"
    next_turn_relevant_sents[qid] = qid_cid
    qid += 1

  return queries, corpus, next_turn_relevant_sents

def prepare_classification_eval_data(eval_data):
  validd = []
  for ed in eval_data:
    gold_ind = ed['next_gold_ind']
    context = ed['corpus']
    history = ed['queries']

    gold_count = sum([1 if l == 1 else 0 for l in gold_ind])
    negative_count = sum([1 if l == 0 else 0 for l in gold_ind])
    all_count = min(gold_count, negative_count)

    gold_sents = [context[i] for i in range(len(context)) if gold_ind[i] == 1]
    if all_count != gold_count:
      gold_sents = random.sample(gold_sents, all_count)

    # randomly select a subset of the negative samples
    neg_sents = [context[i] for i in range(len(context)) if gold_ind[i] == 0]
    if all_count != negative_count:
      neg_sents = random.sample(neg_sents, all_count)

    # append the data
    for gsent in gold_sents:
      validd.append([history, gsent, 1])
    for nsent in neg_sents:
      validd.append([history, nsent, 0])


  dev_sentences1 = []
  dev_sentences2 = []
  dev_labels = []

  for sample in validd:
    dev_sentences1.append(sample[0])
    dev_sentences2.append(sample[1])
    dev_labels.append(sample[2])
  return dev_sentences1, dev_sentences2, dev_labels

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

def contrastive_data(episode_inputs, window_size):
  all_data = []
  for episode_id, episode_info in episode_inputs.items():

    turns = episode_info['turns']
    context = episode_info['context']

    # define max_turn to ensure that the next_turn does not exceed the list.
    max_turn = max([int(l) for l in turns.keys()])

    for turn, turn_info in turns.items():
      # turn_info contains gold_sent_indices and dialogue history.
      next_turn = str(int(turn) + 1)
      if int(next_turn) <= max_turn:
        # The relevant sentences of next turn
        gold_ind = turns[next_turn]['gold_sent_indices']

        #Only consider the cases where there is at least one gold and negative sentence.
        gold_count = sum([1 if l == 1 else 0 for l in gold_ind])
        negative_count = sum([1 if l == 0 else 0 for l in gold_ind])
        all_count = min(gold_count, negative_count)

        if all_count >= 1:

          gold_sents = [context[i] for i in range(len(context)) if gold_ind[i] == 1]
          if all_count != gold_count:
            gold_sents = random.sample(gold_sents, all_count)

          # randomly select a subset of the negative samples
          neg_sents = [context[i] for i in range(len(context)) if gold_ind[i] == 0]
          if all_count != negative_count:
            neg_sents = random.sample(neg_sents, all_count)

          # Define the windowed dialogue history
          history_dict = turn_info['dialogue_history']
          ordered_turns = sorted([int(l) for l in list(history_dict.keys())])
          len_ = len(ordered_turns)
          window = max(ordered_turns[-1], window_size)
          selected_keys = [str(l) for l in ordered_turns[len_ - window:]]
          history_dict = {k:history_dict[k] for k in selected_keys}
          history = ' '.join(list(history_dict.values()))

          # append the data
          for gsent in gold_sents:
            all_data.append([history, gsent, 1])
          for nsent in neg_sents:
            all_data.append([history, nsent, 0])

  return all_data

if __name__ == "__main__":
  with open("../all_data.json", 'r') as read_file:
    episode_inputs = json.load(read_file)

  random.seed(42)


  trainingd, eval_data = preprocess_data(episode_inputs, window_size = 2, jump = 1)
  training_data = []
  print(trainingd[:10])
  history_lengths = []
  corpus_lengths = []
  #model = SentenceTransformer('stsb-distilbert-base')
  #embeddings = []
  model = SentenceTransformer('stsb-roberta-base')
  model.max_seq_length = 128
  #embeddings = []
  for sample in trainingd:
    #import pdb; pdb.set_trace()
    hist = ' '.join(sample[0].split()[-model.max_seq_length:])
    assert len(hist.split()) <= model.max_seq_length, f"history {len(hist.split())} longer than the max_seq{model.max_seq_length}"
    training_data.append(InputExample(texts=[hist, sample[1]], label= sample[2]))
    history_lengths.append(len(sample[0].split()))
    #max_len = max(max_len, len(sample[0].split()))
    corpus_lengths.append(len(sample[1].split()))
    #embedding = model.tokenize(sample[0])
    #print(embedding)
    #embeddings.append(embedding)

  max_len = max(history_lengths)
  mean_len = sum(history_lengths) / len(history_lengths)
  print(f"maximum len (number of words):{max_len} and mean: {mean_len}")
  #import pdb; pdb.set_trace()
  max_len_c = max(corpus_lengths)
  mean_len_c = sum(corpus_lengths) / len(corpus_lengths)
  print(f"maximum corpus len (number of words):{max_len_c} and mean: {mean_len_c}")
  evaluators = []


  queries, corpus, next_turn_relevant_sents = prepare_ir_eval_data(eval_data)
  ir_evaluator = evaluation.InformationRetrievalEvaluator(queries, corpus, next_turn_relevant_sents)
  evaluators.append(ir_evaluator)

  dev_sentences1, dev_sentences2, dev_labels = prepare_classification_eval_data(eval_data)
  binary_acc_evaluator = evaluation.BinaryClassificationEvaluator(dev_sentences1, dev_sentences2, dev_labels)
  evaluators.append(binary_acc_evaluator)
  
  train_dataset = SentencesDataset(training_data, model=model)
  train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=52)
  distance_metric = losses.SiameseDistanceMetric.COSINE_DISTANCE
  margin = 0.5
  train_loss = losses.OnlineContrastiveLoss(model=model, distance_metric=distance_metric, margin=margin)
  seq_evaluator = evaluation.SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[-1])
  num_epochs = 4
  model_save_path = './output/texp_6_roberta_base_evalclass_64/'
  # Train the model
  model.fit(train_objectives=[(train_dataloader, train_loss)],
            evaluator=seq_evaluator,
            epochs=num_epochs,
            warmup_steps=1000,
            output_path=model_save_path
            )
