import json
import math
from nltk.tokenize import sent_tokenize
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm

with open('../../archive/npr_kg_dialogue_lg.jsonl', 'r') as f:
    data = {}
    for l in f:
        data.update(json.loads(l))

with open('../../all_headline_archive_texts.json', 'r') as f:
    meta = json.load(f)

# Parse each dialogue and format the reranker inputs
# For each turn, we extract the dialogue history + the relevant sentence
# indices for this turn.

# Below, we extract the following:
#   episode_context: full list of sentence tokenized articles associated with the k-th turn
#   gold_sent_indices: one-hot encoded relevant sents in episode_context list
#   dialogue_history: the 0:k-1 turns of dialogue, for the k-th turn

episode_inputs = {}
for episode_id, episode_turns in tqdm(data.items()):
    if len(episode_inputs) % 1000 == 0:
        print(len(episode_inputs))

    episode_inputs.update({episode_id: {}})
    episode_context = {}
    # Here, we accumulate all relevant articles for the entire dialogue
    # by looping through each turn and extracting the relevant set
    for k,v in episode_turns['kg_refs'].items():
        for ref in v:
            # This block mimics the original retrieval preprocessor
            # We first retrieve the reference articles for each turn
            # then we parse, sentence tokenize, and filter short sents
            if ref['id'] not in episode_context:
                ref_context = meta[ref['id']]['body'].lower().replace('\n', ' ')
                ref_context = sent_tokenize(ref_context)
                ref_context = [x for x in ref_context if len(x.split()) > 5]
                episode_context[ref['id']] = ref_context

    # Convert dict of article-specific sents to flat list of sents
    episode_context = [v for k,v in episode_context.items()]
    episode_context = [sub for l in episode_context for sub in l]
    print('Extracted {} article sents as episode context'.format(len(episode_context)))

    # We provide the ref sents (context) here that will inform all examples in 'turns'
    episode_inputs[episode_id]['context'] = episode_context
    episode_inputs[episode_id]['turns'] = {}

    # Now, we process the gold_sent_indices based on the full set of article sents
    for k,v in episode_turns['kg_refs'].items():
        # We skip the first turn of dialogue since there is no
        # dialogue history to use as input
        if k == '0': 
            continue
        kg_sents = episode_turns['kg_sents'][k]

        # Construct the dialogue history from the previous turns
        # up to (but not including) the current (kth) turn
        # The exception is the 0th turn, which we skip
        dialogue_history = {int(i):j for i,j in episode_turns['turns'].items() if int(k) > int(i)}

        # Initialize candidate relevance scores to zero
        gold_sent_indices = [0]*len(episode_context)
        if kg_sents:
            # Find the indices for each kg_sent in episode_context and add to gold_sent_indices
            # If we do not have kg_sents for this turn, gold_sent_indices is simple an empty list
            for kg_sent in kg_sents:
                sent_to_ref_index = [n for n,x in enumerate(episode_context) if kg_sent == x or kg_sent in x]
                if sent_to_ref_index:
                    sent_to_ref_index = sent_to_ref_index[0]
                else:
                    raise Exception('No gold sent found')
                gold_sent_indices[sent_to_ref_index] = 1

        # Filter context sents with zero BLEU-2 score into a third class (-1). This
        # helps us split the context sents into a set of quasi-relevant setns (0)
        # that can be used as contrastive negative examples during training.
        for n,sent in enumerate(episode_context):
            b2 = sentence_bleu([sent], episode_turns['turns'][k].lower(), weights=(0.5,0.5,0,0))
            if math.isclose(b2, 0., abs_tol=1e-2) and gold_sent_indices[n] != 1:
                gold_sent_indices[n] = -1

        episode_inputs[episode_id]['turns'][k] = {
                'gold_sent_indices': gold_sent_indices,
                'dialogue_history': dialogue_history
            }
