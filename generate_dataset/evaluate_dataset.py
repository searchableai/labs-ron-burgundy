import json
from common import ngram_overlap

from stop_words import get_stop_words
import nltk
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = list(get_stop_words('en'))  # have around 900 stopwords
nltk_words = list(stopwords.words('english'))  # have around 150 stopwords
stop_words.extend(nltk_words)
stop_words = list(set(stop_words))
remove_list = [x.lower() for x in stop_words]



def evaluate(data):
    epi_scores = {}
    epi_scores.update({'accuracy':{}, 'precision':{}})
    epi_em = {}
    epi_em.update({'accuracy':{}, 'precision':{}})
    null_turns = {}
    for k,v in data.items():
        turn_scores_acc = {}
        turn_scores_prec = {}
        turn_em_acc = {}
        turn_em_prec = {}
        num_null_turns = 0
        for n,t in v['turns'].items():
            # If there are no kg sents for this turn, we simply
            # set the accuracy to zero and skip precision
            if not v['kg_sents'][n]:
                turn_scores_acc[n] = 0.
                turn_em_acc[n] = 0.
                num_null_turns += 1
                continue
            try:
                olap = ngram_overlap(v['kg_sents'][n], t, remove_list)
            except Exception as e:
                print(n,t)
                raise Exception('bad input')
            # Take the max olap score across all kg sents
            turn_scores_acc[n] = max([x.score for x in olap])
            turn_scores_prec[n] = max([x.score for x in olap])
            turn_em_acc[n] = max([x.em for x in olap])
            turn_em_prec[n] = max([x.em for x in olap])
        null_turns[k] = num_null_turns/len(v['turns'].keys())
        # If there are no precision scores, kg_sents was None
        # for the entire episode and we will ignore it
        if not len(turn_scores_prec):
            continue
        # Average olap scores across all dialogue turns
        epi_scores['accuracy'][k] = sum(turn_scores_acc.values())/len(turn_scores_acc)*100.
        epi_em['accuracy'][k] = sum(turn_em_acc.values())/len(turn_em_acc)
        epi_scores['precision'][k] = sum(turn_scores_prec.values())/len(turn_scores_prec)*100.
        epi_em['precision'][k] = sum(turn_em_prec.values())/len(turn_em_prec)
    avg_scores_acc = round(sum(epi_scores['accuracy'].values())/len(epi_scores['accuracy']),2)
    avg_em_acc = round(sum(epi_em['accuracy'].values())/len(epi_em['accuracy']),2)
    avg_scores_prec = round(sum(epi_scores['precision'].values())/len(epi_scores['precision']),2)
    avg_em_prec = round(sum(epi_em['precision'].values())/len(epi_em['precision']),2)
    return (avg_scores_acc, avg_scores_prec, avg_em_acc, avg_em_prec)

if __name__ == "__main__":
    fname = '../../archive/npr_kg_dialogue_lg_minturn.jsonl'
    with open(fname, 'r') as f:
        data = {}
        for d in f:
            data.update(json.loads(d))
    print('Processing {} episodes'.format(len(data.items())))
    results = evaluate(data)
