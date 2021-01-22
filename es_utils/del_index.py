import boto3
from elasticsearch import Elasticsearch, RequestsHttpConnection
from elasticsearch_dsl import connections, Index
from requests_aws4auth import AWS4Auth

import boto3
def delete_index(index, host, region):
    """ Create the index with mapping """

    # ES Config Params
    session = boto3.Session()
    credentials = session.get_credentials()
    region = region
    host = host
    port = 443
    awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, 'es')

    connections.create_connection(hosts=[{'host':host,'port':port}], timeout=60, use_ssl=True, verify_certs=True, http_auth=awsauth, connection_class= RequestsHttpConnection)

    # Delete the index
    print(Index(index).get_mapping())
    index = Index(index).delete(ignore=404)
    print(Index(index).get_mapping())



if __name__ == '__main__':
    index = 'npr-articles-lg-20210108'
    region = 'us-east-1'
    host = "search-research-7hrbumpkbk3qkzqjujfgsrkwre.us-east-1.es.amazonaws.com"
    delete_index(index, host, region)
