#!/usr/bin/env python3

import sys
import os
try:
    from pynlpl.formats import folia
except ImportError:
    print("ERROR: PyNLPl not found, please install pynlpl (pip install pynlpl)",file=sys.stderr)
    sys.exit(2)

if len(sys.argv) == 1:
    print("Usage: tsv2folia.py [tsvfile] [[tsvfile]] ..etc..",file=sys.stderr)
    print("Add parameter --rtl for right-to-left languages (arabic, hebrew, farsi, etc)!")
    sys.exit(2)


rtl = False

for filename in sys.argv[1:]:
    if filename == '--rtl':
        rtl = True
        continue
    targetfilename = filename.replace('.tsv','') + '.folia.xml'
    doc = folia.Document(id=os.path.basename(filename.replace('.tsv','')))
    if rtl:
        doc.metadata['direction'] = 'rtl'
    doc.declare(folia.Entity, "https://github.com/proycon/parseme-support/raw/master/parseme-mwe.foliaset.xml")
    text = doc.append(folia.Text)
    sentence = folia.Sentence(doc,generate_id_in=text)
    with open(filename,'r',encoding='utf-8') as f:
        for line in f:
            if line.strip(): #not empty
                fields = line.strip().split('\t')
                space = not (len(fields) > 2 and fields[2] == 'nsp')
                sentence.append(folia.Word, text=fields[1],space=space)
            elif len(sentence) > 0: #empty and we have a sentence to add
                text.append(sentence)
                sentence = folia.Sentence(doc, generate_id_in=text)
    if sentence.count(folia.Word) > 0: #don't forget the very last one
        text.append(sentence)
    doc.save(targetfilename)



