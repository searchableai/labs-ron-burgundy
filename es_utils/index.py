import json

#fname = '/home/ubuntu/dev/datasets/npr/all_headline_texts_v5.json'
fname = '/home/ubuntu/dev/datasets/npr/all_headline_archive_texts.json'
with open(fname, 'r') as f:
    data = json.load(f)

from elasticsearch_dsl import connections, Index
from elasticsearch_dsl import Q, Search
from elasticsearch_dsl.query import Bool, Match

from elasticsearch import Elasticsearch, RequestsHttpConnection

from elasticsearch_dsl import Document, Text, Keyword
from requests_aws4auth import AWS4Auth

from elasticsearch_dsl import analyzer, tokenizer, analysis

import boto3
session = boto3.Session()
credentials = session.get_credentials()
region = 'us-east-1'
host = "search-research-7hrbumpkbk3qkzqjujfgsrkwre.us-east-1.es.amazonaws.com"
port = 443

awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, 'es')

connections.create_connection(hosts=[{'host':host,'port':port}], timeout=60, use_ssl=True, verify_certs=True, http_auth=awsauth, connection_class= RequestsHttpConnection)
print('CONN: ', connections.get_connection().info())

class Doc(Document):
    guid = Keyword(required=True)
    url = Keyword(required=True)
    date = Keyword(required=True)
    title = Text(required=True)
    text = Text(required=True)

# Create the index with mapping
index = 'npr-articles-lg-20210108'
Doc.init(index=index)

print('Indexing...')

# Index wiki articles
n = 0
for k,v in data.items():
    # Uncomment to index only orig articles
    #if 'a' in k:
    #    continue
    if n % 5000 == 0:
        print('processed {} docs'.format(n))
    doc_obj = Doc()#meta={'id':44})
    doc_obj.guid = k
    if isinstance(v, str):
        print(k,v)
    else:
        doc_obj.date=v['date']
        doc_obj.url=v['url']
        doc_obj.title=v['title']
        doc_obj.text=v['body']
        res = doc_obj.save(index=index)
    n += 1
