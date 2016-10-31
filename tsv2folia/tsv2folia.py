#!/usr/bin/env python3

import argparse
import sys
import os
try:
    from pynlpl.formats import folia
except ImportError:
    print("ERROR: PyNLPl not found, please install pynlpl (pip install pynlpl)",file=sys.stderr)
    sys.exit(2)

EMPTY = ['', '_']

POS_SET_URL = "https://github.com/proycon/parseme-support/raw/master/parseme-pos.foliaset.xml"

# The TSV file shouls have the following fields:
# 0: token ID (restarting from 1 at every sentence)
# 1: surface form
# 2: nsp (optional) -> the string 'nsp' if the current token should be appended to the previous without space 
# 3: MWEs (optional) -> the list of MWEs this word belongs to in the format 1[:MWE_type1]; 2[:MWE_type2]; ...
def convert(filename, targetfilename, rtl, lang_set_file):
    doc = folia.Document(id=os.path.basename(filename.replace('.tsv','')))
    if rtl:
        doc.metadata['direction'] = 'rtl'
    doc.metadata['status'] = 'untouched'
    doc.declare(folia.Entity, lang_set_file) #ENTITY-SET definition    
    doc.declare(folia.AnnotationType.POS, set=POS_SET_URL) #POS-SET definition 
    text = doc.append(folia.Text)
    sentence = folia.Sentence(doc,generate_id_in=text)
    mweInfo = {}  # dict: mwe ID -> {type: mweCat, words:[list of words in the curent MWE]}
    with open(filename,'r',encoding='utf-8') as f:
        for line in f:
            if line.strip(): #not empty
                fields = line.strip().split('\t')
                space = not (len(fields) > 2 and fields[2] == 'nsp')
                currentWord = folia.Word(doc, text=fields[1],space=space, generate_id_in=sentence)
                sentence.append(currentWord)
                if len(fields) > 3 :
                    word_mwes = fields[3]
                    if word_mwes not in EMPTY:
                        word_mwes_split = word_mwes.split(';')
                        for wm in word_mwes_split:
                            #wm is either 'i' or 'i:mweCat' where i is an integer 
                            wm_split = [x.strip() for x in wm.split(':')]
                            index = int(wm_split[0])
                            mweCatsAndWords = mweInfo.setdefault(index, {'words':[]})
                            if len(wm_split)>1:
                                mweCatsAndWords['cat'] = wm_split[1]
                            mweCatsAndWords['words'].append(currentWord)             
                if len(fields) > 4:
                    pos = fields[4]
                    if pos not in EMPTY:
                        posAnnot = folia.PosAnnotation(doc, cls=pos, annotator="auto", annotatortype=folia.AnnotatorType.AUTO )
                        currentWord.append( posAnnot )
            elif len(sentence) > 0: #empty and we have a sentence to add
                mwe_list = folia.EntitiesLayer(doc)
                for mweID, mweDetails in mweInfo.items():
                    mweCat = mweDetails['cat']
                    wordsInMwe = mweDetails['words']
                    print('Adding VMWE {}:{}'.format(mweID, mweCat))                                                            
                    print('wordsInMwe: {}'.format([w.text for w in wordsInMwe]))
                    mwe_list.append(folia.Entity, *wordsInMwe, cls=mweCat, annotatortype=folia.AnnotatorType.MANUAL)
                sentence.append(mwe_list)
                text.append(sentence)
                sentence = folia.Sentence(doc, generate_id_in=text)
                mweInfo = {}  # dict: mwe ID -> {type: mweCat, words:[list of words ids]}                
    if sentence.count(folia.Word) > 0: #don't forget the very last one
        text.append(sentence)
    doc.save(targetfilename)

#This function can be called directly by FLAT
def flat_convert(filename, targetfilename, *args, **kwargs):
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

set_path = 'https://github.com/proycon/parseme-support/raw/master/'

set_options = {
    #"parseme": {
    #    'set_file': set_path + "parseme-mwe.foliaset.xml",
    #    'rtl': False
    #}
    "english": {
        'set_file': set_path + "parseme-mwe-en.foliaset.xml",
        'rtl': False
    },
    "farsi": {
        'set_file': set_path + "parseme-mwe-farsi.foliaset.xml",
        'rtl': True
    },
    "germanic": {
        'set_file': set_path + "parseme-mwe-germanic.foliaset.xml",
        'rtl': False
    },       
    "hebrew": {
        'set_file': set_path + "parseme-mwe-hebrew.foliaset.xml",
        'rtl': True
    },
    "lithuanian": {
        'set_file': set_path + "parseme-mwe-lt.foliaset.xml",
        'rtl': False
    },
    "other": {
        'set_file': set_path + "parseme-mwe-other.foliaset.xml",
        'rtl': False
    },
    "romance": {
        'set_file': set_path + "parseme-mwe-romance.foliaset.xml",
        'rtl': False
    },
    "italian": {
        'set_file': set_path + "parseme-mwe-it.foliaset.xml",
        'rtl': False
    },
    "portuguese": {
        'set_file': set_path + "parseme-mwe-pt.foliaset.xml",
        'rtl': False
    },
    "slavic": {
        'set_file': set_path + "parseme-mwe-slavic.foliaset.xml",
        'rtl': False
    },
    "yiddish": {
        'set_file': set_path + "parseme-mwe-yiddish.foliaset.xml",
        'rtl': True
    }
}

parser = argparse.ArgumentParser(description="Convert from TSV to FoLiA XML.")
parser.add_argument("FILE", type=str, help="An input TSV file") #nargs=1
parser.add_argument("--language", type = str.lower, choices = set_options.keys(), help="The input language") #dest="LANG", nargs=1 
 

class Main(object):
    def __init__(self, args):
        self.args = args
    
    def run(self):
        lang_options = set_options[self.args.language]
        filename = self.args.FILE
        targetfilename = filename.replace('.tsv','') + '.folia.xml'
        rtl = lang_options['rtl']
        lang_set_file = lang_options['set_file']        
        convert(filename, targetfilename, rtl, lang_set_file)

#####################################################

if __name__ == "__main__":
    Main(parser.parse_args()).run()
