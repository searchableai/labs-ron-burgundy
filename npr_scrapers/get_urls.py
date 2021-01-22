""" Retrieve article urls based on a set of headlines """

from requests_html import AsyncHTMLSession
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import asyncio
import csv
import urllib
import pyppdf.patch_pyppeteer

num_thr = 16

with open("headlines.csv") as f:
    csv_reader = csv.reader(f)
    data = []
    do_start = False
    for line in csv_reader:
        if do_start:
            data.append(line)
        if line[0] == start_id:
            do_start = True

    search_queries = {row[0]:row[1].replace('"','') for row in data[1:]}
    qids = list(search_queries.keys())
    qurls = [{'query': search_queries[qid]} for qid in qids]
    qurls = [urllib.parse.urlencode(q) for q in qurls]
    qurls = ["https://www.npr.org/search?"+q+"&page=1&refinementList%5Bshows%5D=" for q in qurls]
    qurls = list(dict(zip(qids, qurls)).items())
print('Getting {} urls'.format(len(qurls)))

def batch_list(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

async def fetch(session, url, qid):
    r = await session.get(url)
    await r.html.arender(sleep=3)
    return (qid, r.html)

async def get_data_asynchronous():  
    global qurls
    results = []
    start_time = datetime.now()
    with ThreadPoolExecutor(max_workers=num_thr) as executor:
        with AsyncHTMLSession() as session:
            loop = asyncio.get_event_loop()
            for batch in list(batch_list(qurls, num_thr)):
                batch_results = {}
                tasks = [
                    await loop.run_in_executor(
                        executor,
                        fetch,
                        *(session, url, qid) # Allows us to pass in multiple arguments to `fetch`
                    )
                    for qid, url in batch
                ]

                for response in await asyncio.gather(*tasks):
                    response_id = response[0]
                    response = response[1]
                    s = response.search('results found in')
                    hit = response.find('.ais-InfiniteHits-item', first=True)
                    try:
                        hit_title = hit.find('.title > a', first=True)
                        href = hit_title.attrs.get('href')
                        res = f"https://www.npr.org{href}" if href else ""
                        if response_id not in results:
                            batch_results.update({response_id: res})
                            results.append(response_id)
                    except Exception as e:
                        print(e)
                if len(results) % 1000 <= 15:
                    print('Fetched {} urls in {} s ({}/s)'.format(len(results), (datetime.now()-start_time).total_seconds(), len(results)/(datetime.now()-start_time).total_seconds()))

                with open('headline_articles.jsonl', 'a+') as f:
                    for k,v in batch_results.items():
                        l = json.dumps({k:v})
                        f.write(l+'\n')
    #print(results)
    #print(len(results))

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    future = asyncio.ensure_future(get_data_asynchronous())
    loop.run_until_complete(future)
