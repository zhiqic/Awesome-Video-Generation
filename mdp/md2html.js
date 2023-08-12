
import { marked } from 'marked'
import { readFileSync, writeFileSync } from 'fs'

const mdText = readFileSync('../README.md', 'utf8')

const html = marked(mdText)

writeFileSync('README.html', html)
