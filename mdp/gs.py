#!/usr/local/bin/python3

import gspread
import requests
from bs4 import BeautifulSoup
import urllib.parse as up
from tqdm import tqdm
import sys
import utils
from thefuzz import fuzz
import pandas as pd

MAX_ROW = 200

def fuzzy_in(a: str, b: list[str], thresh: int = 90):
    for i in b:
        if fuzz.partial_ratio(a, i) > thresh:
            return True
    return False

def main():
    gc = gspread.oauth()
    sp = gc.open('Video Generation Survey')

    for year in range(2021, 2023 + 1):
        papers = pd.read_csv(f'{year}.csv.gz', keep_default_na=False).iterrows()
        tb = []
        w = sp.worksheet(str(year))
        ts = [i[0] for i in w.get('B2:B999')]
        next_free_row = len(ts) + 2
        print(f'Next free row: {next_free_row}')
        cnt = 0
        for p in papers:
            p = p[1]
            if not fuzzy_in(p['title'], ts):
                tb.append([year, p['title'], p['arxiv'], '', p['conf'],
                           p['github'], p['website']])
                cnt += 1
        print(f'Added {cnt} papers to {year}')
        w.update(f'A{next_free_row}:G999', tb)
    

def genMarkdown(year: str):
    gc = gspread.oauth()
    wks = gc.open('Video Generation Survey').worksheet(year)

    tb = wks.get_all_values()

    with open('out.md', 'w') as f:

        for r in tqdm(range(1, len(tb))):

            year, title, arxiv, _, conf, git, web, _ = tb[r]

            f.write(f'+ [{title}]({arxiv})')

            if conf == '':
                f.write('\\\n')
            else:
                f.write(f' ({conf})\\\n')
            
            f.write( '  [![arXiv](https://img.shields.io/badge/'
                    f'arXiv-b31b1b.svg)]({arxiv})\n')
            
            if git != '':
                path = up.urlparse(git).path
                f.write( '  [![Star](https://img.shields.io/github/stars'
                        f'{path}.svg?style=social&label=Star)]({git})\n')

            if web != '':
                f.write( '  [![Website](https://img.shields.io/badge/'
                        f'Website-9cf)]({web})\n')

            f.write('\n')

def fillArxivLinks():

    end_row = 196

    gc = gspread.oauth()
    wks = gc.open('Video Generation Survey').worksheet('2021')
    titles = [i[0] for i in wks.get(f'B2:B{end_row}')]
    links = []

    for t in tqdm(titles):
        link = findLink(t)
        if link == '':
            tqdm.write(f'No results found for "{t}"')
            link = 'N/A'
        links.append([link])
    
    wks.update(f'C2:C{end_row}', links)

def findLink(t: str):

    # Filter out any characters that might fuck with arXiv's fragile search system
    t = t.replace('?', '')

    q = ('https://arxiv.org/search/?query={}&searchtype=title&'
         'abstracts=show&order=&size=200')

    r = requests.get(q.format(up.quote_plus(t)))

    s = BeautifulSoup(r.content, 'html5lib')
    ret = s.findAll('p', attrs={'class': 'list-title is-inline-block'})
    ret = [i.findChildren()[0]['href'].strip() for i in ret]

    if len(ret) == 0:
        return ''

    return ret[0]

if __name__ == '__main__':
    main()