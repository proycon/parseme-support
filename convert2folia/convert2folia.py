#!/usr/bin/env python3

import argparse
import sys
import os
from lxml import etree as ElementTree

#sys.path.append(os.path.dirname(__file__))

import convert2folia.dataalign as dataalign # USE THIS LINE ON FLAT SERVER but comment it out when testing locally
#import dataalign # USE THIS LINE WHEN TESTING LOCALLY but do not forget to comment this line out when committing
from convert2folia.dataalign import folia # USE THIS LINE ON FLAT SERVER but comment it out when testing locally
#from dataalign import folia # USE THIS LINE WHEN TESTING LOCALLY but do not forget to comment this line out when committing

POS_SET_URL = "https://github.com/proycon/parseme-support/raw/master/parseme-pos.foliaset.xml"
CATEG_SET_URL = "https://github.com/proycon/parseme-support/raw/master/parseme-mwe-alllanguages2018.foliaset.xml"
XML_CONLLUP_SEP = dataalign.FoliaIterator.XML_CONLLUP_SEP

parser = argparse.ArgumentParser(description="""
        Convert input file format to FoLiA XML format
        (also add info from CoNLL-U files, if available).""")
parser.add_argument("--input", type=str, nargs="+", required=True,
        help="""Path to input files (in FoLiA XML or PARSEME TSV format)""")
parser.add_argument("--lang", choices=sorted(dataalign.LANGS), metavar="LANG", required=True,
        help="""Name of the target language (e.g. EN, FR, PL, DE...)""")
#parser.add_argument("--conllu", type=str, nargs="+",
#        help="""Path to parallel input CoNLL files""")
parser.add_argument("--stdout", action="store_true", 
        help="""Output data in stdout""")

#####################################################

#def convert(filename, targetfilename, rtl, lang_set_file):
    # if filename.endswith(".parsemetsv") :
        # newfilename=filename.replace(".parsemetsv","")
    # else:
        # newfilename=filename.replace(".tsv","")
    # doc = folia.Document(id=os.path.basename(newfilename))
    # if rtl:
        # doc.metadata['direction'] = 'rtl'
    # doc.metadata['status'] = 'untouched'
    # doc.declare(folia.Entity, lang_set_file) #ENTITY-SET definition    
    # doc.declare(folia.AnnotationType.POS, set=POS_SET_URL) #POS-SET definition 
    # text = doc.append(folia.Text)

    # with open(filename,'r',encoding='utf-8') as f:
        # for tsv_sentence in tsvlib.iter_tsv_sentences(f):
            # folia_sentence = folia.Sentence(doc, generate_id_in=text)
            # text.append(folia_sentence)

            # for tsv_word in tsv_sentence:
                # folia_word = folia.Word(doc, text=tsv_word.surface, space=(not tsv_word.nsp), generate_id_in=folia_sentence)
                # folia_sentence.append(folia_word)
                # if tsv_word.pos:
                    # folia_word.append(folia.PosAnnotation(doc, cls=tsv_word.pos, annotator="auto", annotatortype=folia.AnnotatorType.AUTO))

            # mwe_infos = tsv_sentence.mwe_infos()
            # if mwe_infos:
                # folia_mwe_list = folia.EntitiesLayer(doc)
                # folia_sentence.append(folia_mwe_list)
                # for mweid, mweinfo in mwe_infos.items():
                    # assert mweinfo.category, "Conversion to FoLiA requires all MWEs to have a category"  # checkme
                    # folia_words = [folia_sentence[i] for i in mweinfo.word_indexes]
                    # folia_mwe_list.append(folia.Entity, *folia_words, cls=mweinfo.category, annotatortype=folia.AnnotatorType.MANUAL)
    # doc.save(targetfilename)

#####################################################

class Main:
    def __init__(self, args):
        self.args = args

#########################

    def run(self, lang_set_file=CATEG_SET_URL, outfile=sys.stdout):
        self.conllu_paths = None #self.args.conllu or dataalign.calculate_conllu_paths(self.args.input)
        doc_id = dataalign.basename_without_ext(self.args.input[0])
        doc = folia.Document(id=doc_id) # if doc_id.isalpha() else "_")
        main_text = doc.add(folia.Text)
        if self.args.lang in dataalign.LANGS_WRITTEN_RTL:
            doc.metadata['direction'] = 'rtl'
        doc.metadata['status'] = 'untouched'
        doc.declare(folia.Entity, set=lang_set_file)
        doc.declare(folia.AnnotationType.POS, set=POS_SET_URL)

        iaf = dataalign.IterAlignedFiles(
            self.args.lang, self.args.input, self.conllu_paths, keep_nvmwes=True, debug=False)
        colnames = iaf.aligned_iterator.main_iterators[0].corpusinfo.colnames
        doc.metadata['conllup-colnames'] = XML_CONLLUP_SEP.join(colnames)

        for tsv_sentence in iaf:
            folia_sentence = main_text.add(folia.Sentence)
            for tsv_w in tsv_sentence.tokens:                
                folia_w = folia_sentence.add(folia.Word, text=tsv_w["FORM"], space=(not tsv_w.nsp))

                # Note we swap "\t" and XML_CONLLUP_SEP, for easier human inspection of <conllup-fields>
                conllup_text = XML_CONLLUP_SEP.join(tsv_w.get(col, "_").replace(XML_CONLLUP_SEP, "\t") for col in colnames)
                foreign = ElementTree.Element('foreign-data')
                foreign.append(ElementTree.Element('conllup-columns', columns=conllup_text))
                folia_w.add(folia.ForeignData, node=foreign)               
                if tsv_w.get("UPOS",dataalign.EMPTY) != dataalign.EMPTY and \
                   tsv_w["UPOS"] == "VERB" :
                    folia_w.append(folia.PosAnnotation(doc, cls="V", 
                                   annotator="auto", 
                                   annotatortype=folia.AnnotatorType.AUTO))

            if tsv_sentence.mweoccurs:
                folia_mwe_list = folia_sentence.add(folia.EntitiesLayer)
                for mweo in tsv_sentence.mweoccurs:
                    mweo.metadata.to_folia(mweo, folia_sentence, folia_mwe_list)

            for keyval in tsv_sentence.kv_pairs:
                if isinstance(keyval, dataalign.CommentMetadata):
                    keyval.to_folia(folia_sentence)
                elif keyval.key != 'global.columns':
                    foreign = ElementTree.Element('foreign-data')
                    foreign.append(ElementTree.Element('kv-pair', key=keyval.key, value=keyval.value))
                    folia_sentence.add(folia.ForeignData, node=foreign)

        if outfile == sys.stdout :
            print(doc.xmlstring())
        else :
            doc.save(outfile)

#####################################################

#This function can be called directly by FLAT
def flat_convert(filename, targetfilename, *args, **kwargs):
    if 'rtl' in kwargs and kwargs['rtl']:
        #rtl=True
        lang = "EN" #any non-RTL language
    else:
        lang = "HE" # any RTL language
    setdefinition = kwargs['flatconfiguration']['annotationfocusset']
    try:
        Main(parser.parse_args(['--input',filename,'--lang', lang])).run(lang_set_file=setdefinition,outfile=targetfilename)
        #convert(filename, targetfilename, rtl, setdefinition)
    except Exception as e:
        return False, e.__class__.__name__ + ': ' + str(e) 
    return True

#set_path = 'https://github.com/proycon/parseme-support/raw/master/'
#def valid_input_file(file_name):
#    if not file_name.endswith('.tsv') and not file_name.endswith('.parsemetsv'):
#        raise argparse.ArgumentTypeError("File extension must be '.tsv' or '.parsemetsv'")
#    return file_name

#####################################################

def main():
    Main(parser.parse_args()).run()

#####################################################

if __name__ == "__main__":
    Main(parser.parse_args()).run()
