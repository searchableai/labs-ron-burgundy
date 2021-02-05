import json
import random
from sentence_transformers import InputExample
from sklearn.model_selection import train_test_split
from sentence_transformers import SentencesDataset, SentenceTransformer, evaluation
from torch.utils.data import DataLoader
from sentence_transformers import losses, util

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
  with open("all_data.json", 'r') as read_file:
    episode_inputs = json.load(read_file)
  all_data = contrastive_data(episode_inputs, window_size = 5)
  # create the training set
  training_ratio = 0.9
  training_count = int(training_ratio * len(all_data))
  trainingd, validd = train_test_split(all_data, test_size=len(all_data) - training_count, random_state=42)
  #sample = InputExample(texts=[history, nsent], label= 0) 
            #training_data.append(sample)
  training_data = []
  for sample in trainingd:
    training_data.append(InputExample(texts=[sample[0], sample[1]], label= sample[2]))

  evaluators = []
  dev_sentences1 = []
  dev_sentences2 = []
  dev_labels = []

  for sample in validd:
    dev_sentences1.append(sample[0])
    dev_sentences2.append(sample[1])
    dev_labels.append(sample[2])

  binary_acc_evaluator = evaluation.BinaryClassificationEvaluator(dev_sentences1, dev_sentences2, dev_labels)
  evaluators.append(binary_acc_evaluator)

  model = SentenceTransformer('stsb-distilbert-base')
  train_dataset = SentencesDataset(training_data, model=model)
  train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=2)
  distance_metric = losses.SiameseDistanceMetric.COSINE_DISTANCE
  margin = 0.5
  train_loss = losses.OnlineContrastiveLoss(model=model, distance_metric=distance_metric, margin=margin)
  seq_evaluator = evaluation.SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[-1])
  num_epochs = 5
  model_save_path = './output/model/'
  # Train the model
  model.fit(train_objectives=[(train_dataloader, train_loss)],
            evaluator=seq_evaluator,
            epochs=num_epochs,
            warmup_steps=1000,
            output_path=model_save_path
            )

