#!/usr/bin/env python3

import sys
import os
try:
    from pynlpl.formats import folia
except ImportError:
    print("ERROR: PyNLPl not found, please install pynlpl (pip install pynlpl)",file=sys.stderr)
    sys.exit(2)


def convert(filename, targetfilename, rtl, set_file):
    doc = folia.Document(id=os.path.basename(filename.replace('.tsv','')))
    if rtl:
        doc.metadata['direction'] = 'rtl'
    doc.metadata['status'] = 'converted'
    doc.declare(folia.Entity, set_file)
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

def flat_convert(filename, targetfilename, *args, **kwargs):
    """This function can be called directly by FLAT"""
    if 'rtl' in kwargs and kwargs['rtl']:
        rtl=True
    else:
        rtl = False
    setdefinition = kwargs['flatconfiguration']['annotationfocusset']
    try:
        convert(filename, targetfilename, rtl, setdefinition)
    except Exception as e:
        return False, e.__class__.__name__ + ': ' + str(e)
    return True

def main():
    if len(sys.argv) == 1:
        print("Usage: tsv2folia.py [tsvfile] [[tsvfile]] ..etc..",file=sys.stderr)
        print("Add parameter --rtl for right-to-left languages (arabic, hebrew, farsi, etc)!")
        print("Add parameter --set=parseme[-xx] for specifying a set file (e.g., --set=parseme-en)!")
        sys.exit(2)

    rtl = False

    set_options = {
        "parseme": "https://github.com/proycon/parseme-support/raw/master/parseme-mwe.foliaset.xml",
        "parseme-en": "https://github.com/proycon/parseme-support/raw/master/parseme-mwe-en.foliaset.xml",
        "parseme-germanic": "https://github.com/proycon/parseme-support/raw/master/parseme-mwe-germanic.foliaset.xml",
        "parseme-it": "https://github.com/proycon/parseme-support/raw/master/parseme-mwe-it.foliaset.xml",
        "parseme-lt": "https://github.com/proycon/parseme-support/raw/master/parseme-mwe-lt.foliaset.xml",
        "parseme-other": "https://github.com/proycon/parseme-support/raw/master/parseme-mwe-other.foliaset.xml",
        "parseme-romance": "https://github.com/proycon/parseme-support/raw/master/parseme-mwe-romance.foliaset.xml",
        "parseme-slavic": "https://github.com/proycon/parseme-support/raw/master/parseme-mwe-slavic.foliaset.xml",
    }

    for filename in sys.argv[1:]:
        set_file = set_options["parseme"]
        if filename == '--rtl':
            rtl = True
            continue
        elif filename.startswith('--set=') or filename.startswith('--config='):
            option = filename[filename.find('=')+1:]
            if option in set_options.keys():
                set_file = set_options[option]
                continue
            else:
                print("Wrong set file '{}'".format(option))
            sys.exit(2)
        targetfilename = filename.replace('.tsv','') + '.folia.xml'
        convert(filename, targetfilename, rtl, set_file)

if __name__ == '__main__':
    main()
