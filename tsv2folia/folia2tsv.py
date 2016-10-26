#! /usr/bin/env python3

import argparse
import sys

try:
    from pynlpl.formats import folia
except ImportError:
    print("ERROR: PyNLPl not found, please install pynlpl (pip install pynlpl)", file=sys.stderr)
    sys.exit(2)


parser = argparse.ArgumentParser(description="""
        Convert from FoLiA XML format to TSV (MWEs in a single column).""")
parser.add_argument("FILE", type=str,
        help="""An input XML file in FoLiA format""")

EMPTY = "_"


class Main(object):
    def __init__(self, args):
        self.args = args
        self.doc = folia.Document(file=self.args.FILE)

    def run(self):
        for text in self.doc:
            for sentence in text:
                word2mweinfo = {}  # dict: word ID -> list of "mweid:category"
                mwe_lists = sentence.layers(folia.EntitiesLayer)
                mwes = [mwe for mwe_list in mwe_lists for mwe in mwe_list]
                for mwe_id, mwe in enumerate(mwes, 1):
                    for index_in_mwe, word in enumerate(mwe):
                        mweinfo = word2mweinfo.setdefault(word.id, [])
                        m = "{}:{}".format(mwe_id, mwe.cls) \
                                if index_in_mwe == 0 else str(mwe_id)
                        mweinfo.append(m)

                for i, word in enumerate(sentence.words(), 1):
                    surface_form = word.text() or EMPTY
                    nsp = EMPTY if word.space else "nsp"
                    mwe_ids = ";".join(word2mweinfo.get(word.id, EMPTY))
                    print(i, surface_form, nsp, mwe_ids, sep="\t")
                print()


#####################################################

if __name__ == "__main__":
    Main(parser.parse_args()).run()
