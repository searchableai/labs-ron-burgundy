import re
import json
import csv
import gc
import nltk
import boto3
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import List, Dict, Tuple
from string import punctuation
from common import ngram_overlap
from nltk.tokenize import sent_tokenize
from dateutil.parser import parse

from elasticsearch_dsl import connections, Index
from elasticsearch_dsl import Q, Search
from elasticsearch_dsl.query import Bool, Match
from elasticsearch import Elasticsearch, RequestsHttpConnection
from elasticsearch_dsl import Document, Text, Keyword
from requests_aws4auth import AWS4Auth
from elasticsearch_dsl import analyzer, tokenizer, analysis

from stop_words import get_stop_words
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = list(get_stop_words('en'))  # have around 900 stopwords
nltk_words = list(stopwords.words('english'))  # have around 150 stopwords
stop_words.extend(nltk_words)
stop_words = list(set(stop_words))
remove_list = [x.lower() for x in stop_words]

def preprocess_data(restart: bool=False, outfile=None) -> Tuple[Dict, Dict]:

    if restart and not outfile:
        raise Exception('Outfile param must be specified for restart')

    # Read metadata file for NPR articles
    with open('../all_headline_archive_texts.json', 'r') as f:
        texts = json.load(f)
        article_meta = {k:v['date'] for k,v in texts.items()}

    # Read 2sp dialogues
    with open("utterances-2sp.csv") as f:
        csv_reader = csv.reader(f)
        data = []
        for l in csv_reader:
            data.append(l)
        data = data[1:]
        data = [{'id':x[0],'turn_id':x[1], 'text':x[6], 'is_host': x[5]} for x in data]

    # Load episode title and date
    with open("episodes.csv") as f:
        csv_reader = csv.reader(f)
        episode_data = []
        for l in csv_reader:
            episode_data.append(l)
        episode_data = episode_data[1:]
        episode_data = {x[0]:{'title':x[2], 'date':parse(x[-1])} for x in episode_data}

    # Update data with episode metadata
    for n,d in enumerate(data):
        if d['id'] in episode_data.keys():
            data[n].update({'title':episode_data[d['id']]['title'], 'date':episode_data[d['id']]['date']})

    # Process dialogue and group by ID. Concat all utterances within dialogue ID.
    epi_texts = {}
    context = ''
    current_id = None
    turns = []
    date = None
    for d in data:
        if not current_id:
            current_id = d['id']
        elif current_id and d['id'] != current_id:
            epi_texts[current_id] = {'title':episode_data[current_id]['title'], 'turns':turns, 'date':episode_data[current_id]['date']}
            context = ''
            turns = []
            current_id = d['id']
        context += ' '+d['text']
        turns.append(d['text'])
    if current_id not in epi_texts and context:
        epi_texts[current_id] = {'title':episode_data[current_id]['title'], 'turns':turns, 'date':episode_data[current_id]['date']}

    if restart:
        try:
            with open(outfile, 'r') as f:
                restart_keys = []
                for n in f:
                    n = json.loads(n)
                    restart_keys.append(list(n.keys())[0])
        except Exception as e:
            raise Exception(e)
        for k in restart_keys:
            del epi_texts[k]

        print('Restarting from {} dialogues'.format(len(restart_keys)))

    return epi_texts, article_meta

def _text_to_keywords(text: str):
    nostop_context = [x for x in word_tokenize(text.lower()) if x not in remove_list]
    nostop_context = [x for x in nostop_context if not any(y in x for y in punctuation)]
    toks = list(set(nostop_context))
    return toks

# Block to query elasticsearch and collect results

def _es_search(query: List, dt_episode, meta: Dict, es_index='npr-articles-lg-20210108', top_k: int=4):

    # Number of (non)-unique results to return per query
    es_k = top_k*5
    query_str = ' '.join(query)

    # If the query is an empty string, the input was
    # likely a set of stopwords or non-alpha tokens
    if not query_str:
        return []

    # Weigh title field 10x text field in relevance
    query = Q('query_string', query=query_str, fields=['title^2', 'text^1'])
    s = Search(index=es_index).source(['guid', 'title', 'text']).query(query)[:es_k]
    s = s.highlight('title')
    s = s.highlight('text')
    try:
        res = s.execute()
        res = res.to_dict()
    except Exception as e:
        raise Exception(e)
    titles = []
    top_k_results = [x for x in res['hits']['hits']]
    res = []
    for doc in top_k_results:
        if doc['_source']['title'] not in titles:
            res.append({'id': doc['_source']['guid'], 'text':doc['_source']['text'], 'score':doc['_score'], 'title':doc['_source']['title']})
            titles.append(doc['_source']['title'])

    # Remove articles that are more than 1 year older,
    # or published in a more recent year than the episode
    time_filter_res = []
    for doc in res:
        dt_doc = parse(meta[doc['id']])
        if relativedelta(dt_episode, dt_doc).years in [1, 0]:
            time_filter_res.append(doc)

    mean_score = [x['score'] for x in res]
    mean_score = sum(mean_score)/len(mean_score)

    res = time_filter_res
    res = [x for x in res if x['score'] > mean_score]
    res = res[:top_k]
    return res

def postprocess_data(epi_texts):
    # Remove dialogues that do not contain substantive turns
    empty_indices = []
    for k,v in epi_texts.items():
        if not any([turn for n,turn in enumerate(v['turns']) if len(turn.split())>10]):
            empty_indices.append(k)
    print('Removing {} empty dialogues'.format(len(empty_indices)))
    for i in empty_indices:
        del epi_texts[i]
    return epi_texts

def search_results(turns: List, dt_episode, meta: Dict, min_turn_toks: int=0):
    turn_history = 5
    # Collect ES results for each dialogue in epi_texts
    print('\tGathering es_search results for dialogue {}'.format(k))

    # Concat query terms in single string
    turn_results = {}
    for n,turn in enumerate(turns):
        # Collect turns within a previous window including the current turn
        q_concat_prev = turns[:n+1]
        # Collect turns within a future window including the current turn
        q_concat_curr = turns[n+1:]
        # Omit turns with less than 10 tokens from each dialogue set
        q_concat_prev = [x for x in q_concat_prev if len(x.split())>min_turn_toks]
        q_concat_curr = [x for x in q_concat_curr if len(x.split())>min_turn_toks]
        # Only consider turns with more than 5 tokens as search context
        q_concat = q_concat_prev[-turn_history:]+q_concat_curr[:turn_history]
        q_context = q_concat[:turn_history]
        # Create a set of keyword terms
        q_context = ' '.join(q_context).lower().split()

        # Clean query tokens.
        # If we want to sent tokenize/weight by turn, need to keep periods here
        q_context = [re.sub(r'[^A-Za-z0-9\s\']', '', x) for x in q_context]
        q_context = list(set([x.replace('\'','') for x in q_context if x not in remove_list]))
        q_context = [x for x in q_context if x]

        res = _es_search(q_context, dt_episode, meta)
        turn_results[n] = {'refs': res, 'query': q_context}
    return turn_results

#epi_texts[k]['date'] = epi_texts[k]['date'].strftime('%Y/%m/%d')

def rerank_sents(turns: List, turn_refs: Dict, top_k: int=4, min_turn_toks: int=0, score_thre: float=0.02):
    epi_turns = []
    epi_sents = []
    turn_queries = []
    for n,turn in enumerate(turns):
        sentences = [x['text'].replace('\n', ' ') for x in turn_refs[n]['refs']]
        scores = [x['score'] for x in turn_refs[n]['refs']]
        sentences = [sent_tokenize(x) for x in sentences]
        sentences = [y for x in sentences for y in x]
        # Exclude knowledge sents with less than 5 tokens
        sentences = [x.lower() for x in sentences if len(x.split())>5]
        total_num_sents = len(sentences)

        # Flatten current turn into keyword list. Sent tokenize first?
        query = [x for x in turn.lower().strip().replace('"','').rstrip('.').split()] # if x not in remove_list]
        turn_queries.append(query)

        if not total_num_sents:
            epi_sents.append([])
            continue

        # Ignore kg sents if current turn has less than min toks
        if len(query) < min_turn_toks:
            sorted_answers = []
        else:
            answers = ngram_overlap(sentences, ' '.join(query), remove_list)
            sorted_answers = sorted(answers, key=lambda x: x[1], reverse=True)
            sorted_answers = [(sentences[x[0]], x[1]) for x in sorted_answers][:top_k]
            sorted_answers = [x for x in sorted_answers if x[1] > score_thre]
        epi_sents.append(sorted_answers)

    epi_sents = list(zip(turn_queries, epi_sents))
    epi_turns = dict(zip(range(len(epi_sents)), epi_sents))

    kg_sents = {}
    for n,turn in enumerate(turns):
        kg_sents[n] = [x[0] for x in epi_sents[n][1]]
    kg_refs = {}
    kg_queries = {}
    for n, tref in enumerate(turns):
        turn_refs_sub = [{'id':r['id'], 'title':r['title']} for r in turn_refs[n]['refs']]
        kg_refs[n] = turn_refs_sub
        kg_queries[n] = turn_refs[n]['query']
    kg_turns = {}
    for n,turn in enumerate(turns):
        kg_turns[n] = turn
    return kg_turns, kg_sents, kg_refs, kg_queries

if __name__ == "__main__":
    session = boto3.Session()
    credentials =session.get_credentials()
    region = 'us-east-1'
    configs = {
        'host': "search-research-7hrbumpkbk3qkzqjujfgsrkwre.us-east-1.es.amazonaws.com",
        'port': 443,
        'awsauth': AWS4Auth(credentials.access_key, credentials.secret_key, region, 'es')
    }
    connections.create_connection(hosts=[{'host':configs['host'],'port':configs['port']}], timeout=60, use_ssl=True, verify_certs=True, http_auth=configs['awsauth'], connection_class=RequestsHttpConnection)

    outfile = 'npr_kg_dialogue_lg.jsonl'

    data, meta = preprocess_data(restart=False, outfile=outfile)
    data = postprocess_data(data)
    #Uncomment to select single episode for testing
    #data = {'1':data['1']}
    gc.collect()
    print('Extracted {} dialogues'.format(len(data)))

    search_timer = 0
    rerank_timer = 0
    episode_counter = 0
    for k,v in data.items():
        episode_counter += 1
        start_time = datetime.now()
        turn_refs = search_results(v['turns'], v['date'], meta)
        search_timer += (datetime.now()-start_time).total_seconds()

        start_time = datetime.now()
        kg_turns, kg_sents, kg_refs, kg_queries = rerank_sents(v['turns'], turn_refs)
        rerank_timer += (datetime.now()-start_time).total_seconds()
        

        with open(outfile, 'a+') as f:
            jl = json.dumps({k:{'turns':kg_turns, 'kg_sents':kg_sents, 'kg_refs':kg_refs, 'kg_queries':kg_queries}})
            f.write(jl+'\n')
            f.flush()
        gc.collect()

        print('Avg search time: {} s\tAvg rerank time: {} s'.format(search_timer/episode_counter, rerank_timer/episode_counter))
