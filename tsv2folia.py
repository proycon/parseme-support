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
    sys.exit(2)


for filename in sys.argv[1:]:
    targetfilename = filename.replace('.tsv','') + '.folia.xml'
    doc = folia.Document(id=os.path.basename(filename.replace('.tsv','')))
    doc.declare(folia.Entity, "https://github.com/proycon/parseme-support/raw/master/parseme-mwe.foliaset.xml")
    text = doc.append(folia.Text)
    sentence = text.append(folia.Sentence)
    with open(filename,'r',encoding='utf-8') as f:
        for line in f:
            if line.strip(): #not empty
                fields = line.strip().split('\t')
                space = not (len(fields) >= 2 and fields[2] == 'nsp')
                sentence.append(folia.Word, text=fields[1],space=space)
            elif len(sentence) > 0: #empty and we have a sentence to add
                text.append(sentence)
                sentence = text.append(folia.Sentence)
    if len(sentence) > 0: #don't forget the very last one
        text.append(sentence)
        sentence = text.append(folia.Sentence)
    doc.save(targetfilename)



