import json
import nltk
from common import ngram_overlap
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
nltk.download('wordnet')
rougeL_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

from stop_words import get_stop_words
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = list(get_stop_words('en'))  # have around 900 stopwords
nltk_words = list(stopwords.words('english'))  # have around 150 stopwords
stop_words.extend(nltk_words)
stop_words = list(set(stop_words))
remove_list = [x.lower() for x in stop_words]



def evaluate(data):
    # Accuracy and precision of the ngram_overlap "score", which is
    # a weighted average of ngrams, with non-unigrams weighted
    # more heavily. Precision reflects the overlap between turns
    # and kg_sents where kg_sents are present. Accuracy assigns
    # turns without kg_sents a score of zero.
    epi_scores = {}
    epi_scores.update({'accuracy':{}, 'precision':{}})

    # Accuracy and precision of the exact match (em) ngram score
    # which is the average number of ngrams matches divided by
    # the total possible of ngram matches in each turn.
    epi_em = {}
    epi_em.update({'accuracy':{}, 'precision':{}})

    epi_b1 = {}
    epi_b2 = {}
    epi_b3 = {}
    epi_b4 = {}
    epi_m = {}
    epi_rl = {}

    # Record the average number of turns that do not contain kg_sents
    null_turns = {}

    for k,v in data.items():
        turn_scores_acc = {}
        turn_scores_prec = {}
        turn_em_acc = {}
        turn_em_prec = {}
        num_null_turns = 0
        b1 = 0
        b2 = 0
        b3 = 0
        b4 = 0
        m = 0
        rl = 0
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

                b1 += max([sentence_bleu([x], t, weights=(1, 0, 0, 0)) for x in v['kg_sents'][n]])
                b2 += max([sentence_bleu([x], t, weights=(0.5,0.5,0,0)) for x in v['kg_sents'][n]])
                b3 += max([sentence_bleu([x], t, weights=(0.3333,0.3333,0.3333,0)) for x in v['kg_sents'][n]])
                b4 += max([sentence_bleu([x], t, weights=(0.25,0.25,0.25,0.25)) for x in v['kg_sents'][n]])
                m += max([meteor_score([x], t) for x in v['kg_sents'][n]])
                rl += max([rougeL_scorer.score(x, t)["rougeL"][2] for x in v['kg_sents'][n]])
            except Exception as e:
                print(n,t)
                raise Exception('bad input')
            # Take the max olap score across all kg sents
            turn_scores_acc[n] = max([x.score for x in olap])
            turn_scores_prec[n] = max([x.score for x in olap])
            turn_em_acc[n] = max([x.em for x in olap])
            turn_em_prec[n] = max([x.em for x in olap])

        epi_b1[k] = b1 / len(v['turns'].keys())
        epi_b2[k] = b2 / len(v['turns'].keys())
        epi_b3[k] = b3 / len(v['turns'].keys())
        epi_b4[k] = b4 / len(v['turns'].keys())
        epi_m[k] = m / len(v['turns'].keys())
        epi_rl[k] = rl / len(v['turns'].keys())
    
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
    avg_b1 = round(sum(epi_b1.values()) / len(epi_b1.values()), 2)
    avg_b2 = round(sum(epi_b2.values()) / len(epi_b2.values()), 2)
    avg_b3 = round(sum(epi_b3.values()) / len(epi_b3.values()), 2)
    avg_b4 = round(sum(epi_b4.values()) / len(epi_b4.values()), 2)
    avg_m = round(sum(epi_m.values()) / len(epi_m.values()), 2)
    avg_rl = round(sum(epi_rl.values()) / len(epi_rl.values()), 2)
    
    return {'score_acc': avg_scores_acc, 'score_prec': avg_scores_prec, 'em_acc': avg_em_acc, 'em_prec': avg_em_prec, 'b1': avg_b1, 'b2': avg_b2, 'b3': avg_b3, 'b4': avg_b4, 'm': avg_m, 'rl': avg_rl}

if __name__ == "__main__":
    fname = '../../archive/npr_kg_dialogue_sm.jsonl'
    with open(fname, 'r') as f:
        data = {}
        for d in f:
            data.update(json.loads(d))
    print('Processing {} episodes'.format(len(data.items())))
    results = evaluate(data)
    print(results)
