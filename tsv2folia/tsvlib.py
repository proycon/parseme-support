#! /usr/bin/env python3

r"""
    This is a small library for reading and interpreting the parseme TSV format.

    PARSEME TSV files contain these 4 fields:
    * WordID: an integer (starts at 1)
    * Surface: a string
    * NoSpace: "nsp"  -- or EMPTY
    * MWEs: a list of MWE codes (e.g. "1:LVC;3:ID;4:LVC")  -- or EMPTY

    Additionally, a POS optional field (str or EMPTY) may be present.

    EMPTY columns may contain "_" or "".
"""


import collections
import sys

EMPTY = ["_", ""]


class TSVSentence(list):
    r"""A list of Words."""

    def mwe_infos(self):
        r"""Return a dict {mwe_id: MWEInfo} for all MWEs in this sentence."""
        if self: global_last_lineno(self[0].lineno)
        mwe_infos = {}
        for word_index, word in enumerate(self):
            for mwe_id, mwe_categ in word.mwes_id_categ():
                mwe_info = mwe_infos.setdefault(mwe_id, MWEInfo(mwe_categ, []))
                mwe_info.word_indexes.append(word_index)
        return mwe_infos


class MWEInfo(collections.namedtuple('MWEInfo', 'category word_indexes')):
    r"""Represents all MWEs in a sentence.
    CAREFUL: word indexes start at 0 (not at 1, as in the WordIDs).

    Arguments:
    @type category: str
    @type word_indexes: list[int]
    """
    pass


class TSVWord(collections.namedtuple('Word', 'lineno surface nsp mwe_code pos')):
    r"""Represents a word in the TSV file.

    Arguments:
    @type lineno: int
    @type surface: str
    @type nsp: bool
    @type mwe_code: list[str]
    @type pos: Optional[str]
    """
    def mwes_id_categ(self):
        r"""For each MWE code in `self.mwe_code`, yield an (id, categ) pair.
        @rtype Iterable[(int, Optional[str])]
        """
        global_last_lineno(self.lineno)
        for mwe_str in self.mwe_code:
            split = mwe_str.split(":")
            mwe_id = int(split[0])
            mwe_categ = (split[1] if len(split) > 1 else None)
            yield mwe_id, mwe_categ


def global_last_lineno(lineno):
    # Update global `last_lineno` var
    global last_lineno
    last_lineno = lineno


############################################################


def iter_tsv_sentences(fileobj):
    r"""Yield `TSVSentence` instances for all sentences in the underlying PARSEME TSV file."""
    return TSVReader(fileobj).iter_tsv_sentences()


class TSVReader:
    def __init__(self, fileobj):
        self.fileobj = fileobj


def iter_tsv_sentences(fileobj):
    global last_filename
    last_filename = fileobj.name

    n_fields = len(fileobj.buffer.peek().split(b"\n")[0].split(b"\t"))
    if 3 <= n_fields <= 5:
        return iter_tsv_sentences_official(fileobj)
    elif 8 <= n_fields:
        return iter_tsv_sentences_platinum(fileobj)
    else:
        raise Exception("Bad input file: header does not match a PARSEME TSV format")


def iter_tsv_sentences_official(fileobj):
    # Format: rank|token|nsp|mwe-codes|pos
    sentence = TSVSentence()
    for lineno, line in enumerate(fileobj, 1):
        global_last_lineno(lineno)
        if line.strip():
            fields = line.strip().split('\t')
            fields.extend([""]*5)  # fill in the optional fields
            surface = fields[1]
            nsp = (fields[2] == 'nsp')
            mwe_codes = [] if fields[3] in EMPTY else fields[3].strip().split(";")
            pos = None if fields[4] in EMPTY else fields[4]
            sentence.append(TSVWord(lineno, surface, nsp, mwe_codes, pos))
        else:
            yield sentence
            sentence = TSVSentence()
    if sentence:
        yield sentence


def iter_tsv_sentences_platinum(fileobj):
    # Format: rank|token|nsp|mtw|1st-mwe-id|1st-type|2nd-mwe-id|2nd-type|[3rd...Nth]|[comments]
    next(fileobj); next(fileobj)  # skip the 2-line header
    sentence = TSVSentence()
    for lineno, line in enumerate(fileobj, 1):
        global_last_lineno(lineno)
        if line.strip():
            fields = line.strip().split('\t')
            fields.extend([""]*9)  # fill in the optional fields
            surface = fields[1]
            nsp = (fields[2] == 'nsp')
            # Ignore MTW in fields[3]
            mwe_codes = ["{}:{}".format(fields[i], fields[i+1])
                    for i in xrange(4, len(fields)-1, 2) if fields[i] not in EMPTY]
            # Ignore free comments in fields[-1], present if len(fields)%2==1
            sentence.append(TSVWord(lineno, surface, nsp, mwe_codes, None))
        else:
            yield sentence
            sentence = TSVSentence()
    if sentence:
        yield sentence


#####################################################################

def excepthook(exctype, value, tb):
    global last_lineno
    global last_filename
    if value and last_lineno:
        last_filename = last_filename or "???"
        err_msg = "===> ERROR when reading {} (line {})" \
                .format(last_filename, last_lineno)
        if sys.stderr.isatty():
            err_msg = "\x1b[31m{}\x1b[m".format(err_msg)
        print(err_msg, file=sys.stderr)
    return sys.__excepthook__(exctype, value, tb)


#####################################################################

if __name__ == "__main__":
    sys.excepthook = excepthook
    with open(sys.argv[1]) as f:
        for tsv_sentence in iter_tsv_sentences(f):
            print("TSVSentence:", tsv_sentence)
            print("MWEs:", tsv_sentence.mwe_infos())
