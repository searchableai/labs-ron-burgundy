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

credentials = boto3.Session().get_credentials()

service = "es"
awsauth = AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            region,
            service,
            session_token=credentials.token)
client = Elasticsearch(
            hosts            = [{"host": host, "port": port}],
            use_ssl          = True,
            verify_certs     = True,
            http_auth        = awsauth,
            connection_class = RequestsHttpConnection)
res = client.count(index='npr-articles-20210108')["count"]
print(res)
