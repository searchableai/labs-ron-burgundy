from requests_html import AsyncHTMLSession
from concurrent.futures import ThreadPoolExecutor
import pyppdf.patch_pyppeteer
import asyncio
import csv
import json
import urllib
import pickle


num_thr = 40

#with open("all_url.json") as f:
#with open("remaining_urls.json") as f:
#with open("new_headlines.json", 'r') as f:
#with open('new_urls_v2.json', 'r') as f:
#with open('missing_urls.json', 'r') as f:
with open('archive_urls.json', 'r') as f:
    data = json.load(f)
    qids = [one for one in data]
    qurls = [data[one] for one in data]
    for n,url in enumerate(qurls):
        if 'wbur.org' in url:
            qurls[n] = 'http://'+url.split('http://')[1]
    url_dict = dict(zip(qids, qurls))
    qurls = list(zip(qids, qurls))
    #qurls = qurls[20:]
final_urls = qurls

'''
# Uncomment to start from previous url key
start_url = "578410"
start = False
final_urls = []
for url in qurls:
    if start:
        final_urls.append(url)
    if url[0] == start_url:
        start = True
print('Processing {} urls'.format(len(final_urls)))
'''
    
def batch_list(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]
        
async def fetch(session, url, qid):
    # print(i)
    # print(url)
    print(url)
    try:
        r = await session.get(url)
        await r.html.arender(wait=3.)
        r.close()
        return (qid, r.html)
    except Exception as e:
        print(e)
        return (qid, '')

async def get_data_asynchronous():  
    global final_urls
    with ThreadPoolExecutor(max_workers=num_thr) as executor:
        with AsyncHTMLSession() as session:
            loop = asyncio.get_event_loop()
            for batch in list(batch_list(final_urls, num_thr)):
                results = {}
                
                tasks = [
                  await loop.run_in_executor(
                    executor,
                    fetch,
                    *(session, url, qid) # Allows us to pass in multiple arguments to `fetch`
                  )
                  for qid, url in batch
                ]
                for response in await asyncio.gather(*tasks):
                    #breakpoint()
                    response_id = response[0]
                    response = response[1]
                    if not response:
                        continue
                    #s = response.search('results found in')
                    #hit = response.find('.ais-InfiniteHits-item', first=True)
                    date = [one.text for one in response.find(".dateblock")]
                    if date and date[0]:
                        date = date[0]
                    title = [one.text for one in response.find(".storytitle")]
                    body = [one.text for one in response.find("#storytext>p")]
                    #print(response_id, url_dict[response_id], not title, not body)
                    if title and title[0] and any(body):
                        results.update({response_id: {"title":title[0], "body":' '.join(body), 'date': date}})
                with open("archive_url_texts.jsonl", "a+") as f:
                    for k,v in results.items():
                        l = json.dumps({k:v})
                        f.write(l+'\n')
        
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    future = asyncio.ensure_future(get_data_asynchronous())
    loop.run_until_complete(future)
