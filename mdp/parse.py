
import bs4
from bs4 import BeautifulSoup
import sys, os, pickle, re, pprint
import pandas as pd

def tag_err(expected: str, got: bs4.element.Tag, showGot: bool = False):
    return (f'Line {got.sourceline}, col {got.sourcepos} : '
            f'Expected {expected}') + (f' but got {got.name}' if showGot else '')

os.system('node md2html.js')
with open('README.html', 'r') as f:
    html = f.read()

s = BeautifulSoup(html, 'html5lib')

years = s.find_all('h2', string=re.compile('^\d{4}$'))

out = {}

for y in years:

    ul = y.find_next()

    if ul.name != 'ul':
        print(tag_err('<ul>', ul))
        quit()
    
    out[y.string] = []
    lis = ul.findChildren('li', recursive=False)

    for l in lis:

        a = l.find('p').findChildren('a', recursive=False)

        if a[0].string is None:
            print(tag_err('string', a[0]))
            quit()
            
        for i in a:
            if i.get('href') is None:
                print(tag_err('<a> with href', a))
                quit()

        title = a[0].string
        arxiv = a[0].get('href')
        conf, github, website = [None] * 3

        a0_next = a[0].next_sibling

        if type(a0_next) == bs4.element.NavigableString:
            conf = a0_next.strip()[1:-1]

        if conf is None and a0_next.name != 'br':
            print(tag_err('<br> to follow', a[0]))
            quit()
        
        if conf is not None and len(a[1:]) != 0 and a0_next.next_sibling.name != 'br':
            print(tag_err('<br> to follow', a[0]))
            quit()

        for i in a[1:]:
            img = i.find('img')
            if img is None:
                print(tag_err('enclosing <img>', i))
                quit()
            alt = img.get('alt')
            href = i.get('href')
            if alt == 'Star':
                github = href
            elif alt == 'Website':
                website = href
            elif alt == 'arXiv':
                pass
            else:
                print(tag_err('alt="Star/Website/arXiv"', i))
                quit()

        out[y.string].append({
            'title': title, 'arxiv': arxiv, 'conf': conf,
            'github': github, 'website': website,
        })

#pprint.PrettyPrinter(indent=2).pprint(out)

for y in out.keys():
    df = pd.DataFrame.from_dict(out[y])
    df.to_csv(f'{y}.csv.gz', index=False)
    #pd.options.display.max_colwidth = 15
    #print(df)