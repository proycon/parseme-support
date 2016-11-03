#!/usr/bin/env python3

import tsv2folia.tsvlib
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

    with open(filename,'r',encoding='utf-8') as f:
        for tsv_sentence in tsvlib.iter_tsv_sentences(f):
            folia_sentence = folia.Sentence(doc, generate_id_in=text)
            text.append(folia_sentence)

            for tsv_word in tsv_sentence:
                folia_word = folia.Word(doc, text=tsv_word.surface, space=(not tsv_word.nsp), generate_id_in=folia_sentence)
                folia_sentence.append(folia_word)
                if tsv_word.pos:
                    folia_word.append(folia.PosAnnotation(doc, cls=tsv_word.pos, annotator="auto", annotatortype=folia.AnnotatorType.AUTO))

            mwe_infos = tsv_sentence.mwe_infos()
            if mwe_infos:
                folia_mwe_list = folia.EntitiesLayer(doc)
                folia_sentence.append(folia_mwe_list)
                for mweid, mweinfo in mwe_infos.items():
                    assert mweinfo.category, "Conversion to FoLiA requires all MWEs to have a category"  # checkme
                    folia_words = [folia_sentence[i] for i in mweinfo.word_indexes]
                    folia_mwe_list.append(folia.Entity, *folia_words, cls=mweinfo.category, annotatortype=folia.AnnotatorType.MANUAL)

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
parser.add_argument("--stdout", action="store_true", help="Output data in stdout")
parser.add_argument("--language", type = str.lower, choices = set_options.keys(), help="The input language") #dest="LANG", nargs=1 
 

class Main(object):
    def __init__(self, args):
        self.args = args

    def run(self):
        sys.excepthook = tsvlib.excepthook
        lang_options = set_options[self.args.language]
        filename = self.args.FILE
        targetfilename = "/dev/stdout" if self.args.stdout else filename.replace('.tsv','') + '.folia.xml'
        rtl = lang_options['rtl']
        lang_set_file = lang_options['set_file']        
        convert(filename, targetfilename, rtl, lang_set_file)

#####################################################

if __name__ == "__main__":
    Main(parser.parse_args()).run()
