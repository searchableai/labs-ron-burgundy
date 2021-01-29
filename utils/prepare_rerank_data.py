from nltk.tokenize import sent_tokenize
import json

with open('../archive/npr_kg_dialogue_lg.jsonl', 'r') as f:
    data = {}
    for l in f:
        data.update(json.loads(l))

with open('../all_headline_archive_texts.json', 'r') as f:
    meta = json.load(f)

# Parse each dialogue and format the reranker inputs
# Here, we extract the following:
#   ref_sents: full list of sentence tokenized articles associated with each turn
#   gold_sent_indices: one-hot encoded relevant sents in ref_sents list
#   dialogue_history: the 0:k turns of dialogue, for the kth turn inputs 

episode_contexts = {}
for episode_id, episode_turns in data.items():
    if len(episode_contexts) % 1000 == 0:
        print(len(episode_contexts))

    episode_contexts.update({episode_id: {}})
    for k,v in episode_turns['kg_refs'].items():
        turn_context = []
        for ref in v:
            # This block mimics the original retrieval preprocessor
            # We first retrieve the reference articles for each turn
            # then we parse, sentence tokenize, and filter short sents
            ref_context = meta[ref['id']]['body'].lower().replace('\n', ' ')
            ref_context = sent_tokenize(ref_context)
            ref_context = [x for x in ref_context if len(x.split()) > 5]
            turn_context += ref_context
        kg_sents = episode_turns['kg_sents'][k]

        # Construct the dialogue history from the previous turns up to the current (kth) turn
        dialogue_history = {int(i):j for i,j in episode_turns['turns'].items() if int(k) >= int(i)}

        # Initialize candidate relevance scores to zero
        gold_sent_indices = [0]*len(turn_context)
        if kg_sents:
            # Find the indices for each kg_sent in turn_context and add to gold_sent_indices
            # If we do not have kg_sents for this turn, gold_sent_indices is simple an empty list
            for kg_sent in kg_sents:
                sent_to_ref_index = [n for n,x in enumerate(turn_context) if kg_sent == x or kg_sent in x]
                if sent_to_ref_index:
                    sent_to_ref_index = sent_to_ref_index[0]
                else:
                    print(episode_id, k, '\nKG Sent: ', kg_sent, '\nTurn: ', episode_turns['turns'][k], '\nTurn Context: ', turn_context)
                    raise Exception('No gold sent found')
                gold_sent_indices[sent_to_ref_index] = 1

        episode_contexts[episode_id][k] = {
                'ref_sents': turn_context,
                'gold_sent_indices': gold_sent_indices,
                'dialogue_history': dialogue_history
            }
