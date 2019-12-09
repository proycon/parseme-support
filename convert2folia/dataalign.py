#! /usr/bin/env python3

r"""
This is a library for reading FoLiA XML files aligned with a CoNLL file.
The XML files may have PARSEME annotations, and this library can be helpful
in dealing with such data.

This library requires PyNLPl to be installed.

-----------------------------------------------

To adapt this library to a new language, see the LANGS
and other variables below. You may also want to add special code for
a canonical form (used e.g. when grouping MWEs during consistency check):

  * To fix word lemmas, add special code to `_with_fixed_tokens`
    (e.g. to re-lemmatize "me" and "te" as 3rd-person "se" in Romance languages).

  * To change word order, add special code to `_with_reordered_tokens`.
    (e.g. to reorder the LVC "(a) bath (was) taken" as "take bath").
"""


import abc
import collections
import difflib
import itertools
import json
import os
import re
import sys


# Import the Categories class
#from categories import Categories # USE THIS LINE WHEN TESTING LOCALLY but do not forget to comment this line out when committing
from convert2folia.categories import Categories # USE THIS LINE ON FLAT SERVER but comment it out when testing locally


try:
    from pynlpl.formats import folia
except ImportError:
    exit("ERROR: FoliaPY not found, please run this code: pip3 install folia")


# The `empty` field in CoNLL-U and PARSEME-TSV
EMPTY = "_"

# Languages where the canonical form should have the lemmas for all tokens
# Reason: HI = has many MVCs; HU = has bad POS tags
# (XXX this is a workaround, we should rethink this for ST 2.0)
LANGS_WITH_ALL_CANONICAL_TOKENS_LEMATIZED = set("HI HU".split())


############################################################

# Set of all valid languages in the latest PARSEME Shared-Task
LANGS = set("AR BG CS DE EL EN ES EU FA FR HE HR HU HI IT LT MT PL PT RO SL SV TR".split())

# Languages where the pronoun in IRV is canonically on the left
LANGS_WITH_CANONICAL_REFL_PRON_ON_LEFT = set("DE EU FR RO".split())

# Languages where the verb canonically appears to the right of the object complement (SOV/OSV/OVS)
LANGS_WITH_CANONICAL_VERB_ON_RIGHT = set("DE EU HI TR".split())

# Languages where the verb occurrences usually appear to the right of the object complement (SOV/OSV/OVS)
LANGS_WITH_VERB_OCCURRENCES_ON_RIGHT = LANGS_WITH_CANONICAL_VERB_ON_RIGHT - set(["DE"])

# Languages that are written right-to-left (FLAT needs to know this for proper displaying)
LANGS_WRITTEN_RTL = set(["FA HE YI"])


############################################################
def interpret_color_request(stream, color_req: str) -> bool:
    r"""Interpret environment variables COLOR_STDOUT and COLOR_STDERR ("always/never/auto")."""
    return color_req == 'always' or (color_req == 'auto' and stream.isatty())

# Flags indicating whether we want to use colors when writing to stderr/stdout
COLOR_STDOUT = interpret_color_request(sys.stdout, os.getenv('COLOR_STDOUT', 'auto'))
COLOR_STDERR = interpret_color_request(sys.stderr, os.getenv('COLOR_STDERR', 'auto'))


############################################################

class KVPair:
    r"""Key-value pair for CoNLL-UP: `# <key> = <value>`.
    May represent more elaborate metadata (see Metadata and subclasses).
    """
    GLOBAL_COLUMNS_REGEX = re.compile(rb'^# *global\.columns *= *(.*)$', re.MULTILINE)
    KV_REGEX = re.compile(r'^# *(\S+) *= *(.*?) *$')

    def __init__(self, key: str, value: str):
        self._key, self._value = key, value

    @property
    def key(self):
        return self._key

    @property
    def value(self):
        return self._value

    def __repr__(self):
        return '{}(key={!r}, value={!r})'.format(type(self).__name__, self.key, self.value)

    def to_tsv(self) -> str:
        r"""Return key-value line in CoNLL-UP syntax."""
        if self.key == "RAWLINE":
            return "# {}".format(self.value).rstrip(" ")
        return "# {} = {}".format(self.key, self.value)

    @staticmethod
    def from_conllup(key: str, value: str):
        r"""Return a KVPair for given line of CoNLL-UP."""
        if key == "metadata":
            return json.loads(value, object_hook=KVPair.object_hook)
        return KVPair(key, value)

    @staticmethod
    def object_hook(obj):
        r"""JSON object hook to recursively convert objects
        to Metadata instances when appropriate, and OrderedDict otherwise."""
        obj = collections.OrderedDict(obj)
        if "kind" in obj:
            return Metadata.from_dict(obj)
        return obj


class Metadata(KVPair, abc.ABC):
    r"""Represents common metadata from user annotations
    (This is basically the set of key-value properties
    that are always acceptable in all FoLiA nodes).
    Appears as "# metadata = ..." in CoNLL-UP files.

    Attributes:
    * kind: Type of metadata (e.g. "comment" if CommentMetadata)
    * annotator: Name or ID of the system/human annotator (or None).
    * annotatortype: Either "manual" or "auto" (or None).
    * confidence: Floating point value between zero and one (or None).
    * datetime: Date in format "YYYY-MM-DDThh:mm:ss" (or None).
    * nested: A list of `CommentMetadata`s
    """
    TSV_REGEX = re.compile(r'^# *metadata *= *(.*)$')

    def __init__(self, kind: str, *,
                 annotator: str = None, annotatortype: str = None,
                 datetime: str = None, confidence: float = None,
                 nested: list = None, **kwargs):
        super().__init__(None, None)
        self.kind = kind
        self.annotator = annotator
        self.annotatortype = annotatortype
        self.datetime = datetime
        self.confidence = confidence
        self.nested = nested or []
        assert all(isinstance(x, Metadata) for x in self.nested), self.nested
        for key in kwargs:
            do_warn("Ignoring unexpected key for {type}: {key!r}",
                    type=type(self).__name__, key=key)


    @staticmethod
    def from_folia(f: folia.AbstractElement):
        r"""Return Metadata for given FoLiA element `f`."""
        return Metadata._instantiate_from_folia(
            f, annotator=f.annotator, annotatortype=f.annotatortype,
            datetime=(f.datetime.isoformat() if f.datetime else None), confidence=f.confidence,
            nested=[Metadata.from_folia(c) for c in f.select(folia.Comment)])

    @staticmethod
    def _instantiate_from_folia(f: folia.AbstractElement, **kwargs):
        if isinstance(f, (folia.Comment, folia.Description)):
            # We convert <desc> tags to <comment>, for simplicity
            return CommentMetadata(f.value, **kwargs)
        elif isinstance(f, folia.Entity):
            return MWEAnnotMetadata(**kwargs)  # [("class", f.cls), ("token_ids", .....)])  <-- useless repetition
        else:
            assert False, f


    @abc.abstractmethod
    def specific_properties(self):
        r"""Return properties that are specific to this kind of metadata."""
        raise NotImplementedError

    def generic_properties(self):
        r"""Return properties that are common to all metadata."""
        return [('annotator', self.annotator),
                ('annotatortype', self.annotatortype),
                ('datetime', self.datetime),
                ('confidence', self.confidence)]

    def _folia_kwargs(self) -> dict:
        r"""Return kwargs for FoLiA instantiation of subclasses."""
        return dict((k, v) for (k, v) in self.generic_properties() if v is not None)


    def is_empty(self):
        r'''True iff all of the properties of this Metadata are empty.'''
        return all(v is None for (k, v) in self.specific_properties()) and self.is_generically_empty()

    def is_generically_empty(self):
        r'''True iff all of the generic properties of this Metadata are empty.'''
        return all(v is None for (k, v) in self.generic_properties()) and not self.nested


    def to_dict(self):
        r"""Return an OrderedDict object that represents
        this metadata (e.g. to be converted to JSON).
        """
        ret = collections.OrderedDict()
        ret["kind"] = self.kind
        ret.update((k, v) for k, v in self.specific_properties() if v is not None)
        ret.update((k, v) for k, v in self.generic_properties() if v is not None)
        if self.nested:
            ret["nested"] = [n.to_dict() for n in self.nested]
        return ret

    @staticmethod
    def from_dict(data_dict):
        r"""Return a Metadata for given dict."""
        kind = data_dict.pop("kind")
        if kind == "comment":
            return CommentMetadata(text=data_dict.pop("text"), **data_dict)
        elif kind == "mweinfo":
            return MWEAnnotMetadata(**data_dict)
        else:
            raise Exception("Unknown metadata kind: {!r}".format(kind))

    @property
    def key(self):
        r"""Satisfy `key` element of base class KVPair"""
        return "metadata"

    @property
    def value(self):
        r"""Satisfy `value` element of base class KVPair"""
        return json.dumps(self.to_dict())


class CommentMetadata(Metadata):
    r"""Metadata representing
    a) A fully-fledged comment (as in FoLiA)
    b) A raw comment (as possible in CoNLL-UP)

    The actual type of comment depends on whether there are other specified metadata.
    """
    def __init__(self, text: str, **kwargs):
        super().__init__(kind="comment", **kwargs)
        self.text = text

    def specific_properties(self):
        return [("text", self.text)]

    def to_folia(self, parent: folia.AbstractElement):
        r"""Construct a `folia.Comment` object representing this comment.
        The object will be appended to `parent`.
        """
        child = parent.append(folia.Comment, value=self.text, **self._folia_kwargs())
        for sub in self.nested:
            sub.to_folia(child)


class MWEAnnotMetadata(Metadata):
    r"""Metadata representing an MWE annotation."""
    def __init__(self, mweid : str = None, **kwargs):
        super().__init__(kind="mweinfo", **kwargs)
        assert isinstance(mweid, (str, type(None))), mweid
        self.mweid = mweid  # type: Optional[str]

    def specific_properties(self):
        if self.mweid:
            return [("mweid", self.mweid)]
        do_warn("BUG: field mweid is missing from metadata")
        return []

    def to_folia(self, mweoccur: 'MWEOccur', sentence: folia.Sentence, layer: folia.EntitiesLayer):
        r"""Construct a `folia.Entity` object representing this comment.
        The object will refer to `sentence`, and will be appended to `layer`.
        """
        assert mweoccur.category, "Conversion to FoLiA requires all MWEs to have a category"  # checkme
        folia_words = [sentence[i] for i in mweoccur.indexes]
        child = layer.append(
            folia.Entity, *folia_words, cls=mweoccur.category,
            **self._folia_kwargs())
        for sub in self.nested:
            sub.to_folia(child)



############################################################

class Token(collections.Mapping):
    r"""Represents a token in an input file.

    Instances behave like a frozen dict, so you can do
    e.g. token["LEMMA"] to obtain the lemma.
    """
    def __init__(self, *args, **kwargs):
        data = dict(*args, **kwargs)
        # (Note we allow FORM=="_", because it can mean underspecified OR "_" itself
        self._data = {str(k): str(v) for (k, v) in data.items()
                      if v and (v != '_' or k == 'FORM')}
        self._data.setdefault('FORM', '_')

    def with_update(self, *args, **kwargs):
        r'''Return a copy Token with updated key-value pairs.'''
        ret = Token(self._data)
        ret._data.update(*args, **kwargs)
        return ret

    def with_nospace(self, no_space: bool):
        r'''Return a copy Token with updated MISC (only if no_space is True).'''
        if no_space and 'SpaceAfter=No' not in self.get('MISC', ''):
            new_misc = 'SpaceAfter=No'
            if self.get('MISC'):
                new_misc = "{}|{}".format(self['MISC'], new_misc)
            return self.with_update(MISC=new_misc)
        return self

    def lemma_or_surface(self):
        r'''Return the lemma, if known, or the surface form otherwise'''
        return self.get('LEMMA', self['FORM'])

    def __iter__(self):
        return iter(self._data)
    def __len__(self):
        return len(self._data)
    def __getitem__(self, key):
        return self._data[key]
    def __hash__(self):
        return hash(frozenset(self.items()))
    def __repr__(self):
        return 'Token({})'.format(self._data)

    def cmp_key(self):
        r'''Deterministic exception-free comparison method.'''
        return (self.rank, self.surface, self.nsp, self.get('LEMMA', ''),
                self.univ_pos, self.get('HEAD'), self.get('DEPREL'))

    @property
    def rank(self):
        return self['ID']
    @property
    def surface(self):
        return self['FORM']
    @property
    def nsp(self):
        return 'SpaceAfter=No' in self.get('MISC', '')
    @property
    def lemma(self):
        return self['LEMMA']
    @property
    def univ_pos(self):
        return self.get('UPOS')

    def has_dependency_info(self):
        return 'DEPREL' in self and 'HEAD' in self


class CorpusInfo:
    r"""Information from the input corpus."""
    def __init__(self, lang: str, file_path: str, colnames: tuple):
        self.lang = lang
        self.file_path = file_path
        self.colnames = colnames

class Sentence:
    r"""A sequence of tokens.
    Each MWE is represented as an MWEOccur (with a MWEAnnotMetadata).
    Other KVPair metadata are stored in kv_pairs.
    """
    def __init__(self, corpusinfo: CorpusInfo, nth_sent: int, lineno: int):
        self.corpusinfo = corpusinfo
        self.nth_sent = nth_sent
        self.lineno = lineno
        self.tokens = []              # type: list[Token]
        self.mweoccurs = []           # type: list[MWEOccur]
        self.kv_pairs = []            # type: list[KVPair]

    @property
    def file_path(self):
        return self.corpusinfo.file_path

    def __str__(self):
        r"""Return a string representation of the tokens"""
        return " ".join(map(lambda x: x.surface, self.tokens))

    def empty(self):
        r"""True iff this Sentence is empty."""
        return not (self.tokens or self.mweoccurs)

    def rank2index(self):
        r"""Return a dictionary mapping string ranks to indexes."""
        return {t.rank: index for (index, t) in enumerate(self.tokens)}

    def tokens_and_mwecodes(self):
        r"""Yield pairs (token, mwecodes) of type (Token, list[str])."""
        tokenindex2mweindex = collections.defaultdict(list)
        for mweindex, mweoccur in enumerate(self.mweoccurs):
            for index in mweoccur.indexes:
                tokenindex2mweindex[index].append(mweindex)

        for itoken, token in enumerate(self.tokens):
            mwe_is = tokenindex2mweindex[itoken]
            yield token, [self._mwecode(itoken, mwe_i) for mwe_i in mwe_is]

    def _mwecode(self, itoken, mwe_i):
        r"""Return a string with mweid:category (or just mweid in some cases)."""
        mweoccur = self.mweoccurs[mwe_i]
        if mweoccur.category and mweoccur.indexes[0] == itoken:
            return "{}:{}".format(mwe_i+1, mweoccur.category)
        return str(mwe_i+1)

    def remove_non_vmwes(self):
        r"""Change the mwe_codes in `self.tokens` so as to remove all NonVMWE tags."""
        self.mweoccurs = [m for m in self.mweoccurs if m.category not in Categories.NON_MWES]

    def remove_duplicate_mwes(self):
        r"""Uniqs self.mweoccurs (keeps only first occurrence)"""
        old_mweoccurs = self.mweoccurs
        self.mweoccurs = [m for (i, m) in enumerate(self.mweoccurs) if m not in self.mweoccurs[:i]]
        if len(self.mweoccurs) != len(old_mweoccurs):
            duplicates = [m for m in old_mweoccurs if old_mweoccurs.count(m) > 1]
            for mweoccurs in duplicates:
                self.warn("Removed duplicate MWE: {}".format(mweoccurs))


    def re_tokenize(self, new_tokens: 'list[Token]', indexmap: 'dict[int,list[int]]'):
        r"""Replace `self.tokens` with given tokens and fix `self.mweoccurs` based on `indexmap`"""
        self_nsps = set(i for (i, t) in enumerate(self.tokens) if t.nsp)
        self.tokens = [t.with_nospace(i in self_nsps) for (i, t) in enumerate(new_tokens)]
        self.mweoccurs = [m.remapped_indexes(indexmap) for m in self.mweoccurs]


    def errprefix(self, *, short=True):
        r"""Return a sentence ID, such as "foo.xml(s.13):78"."""
        ret = self.corpusinfo.file_path if not short else os.path.basename(self.corpusinfo.file_path)
        ret += "(s.{})".format(self.nth_sent) if self.nth_sent else ""
        return ret + (":{}".format(self.lineno) if self.lineno else "")

    def warn(self, msg_fmt, **kwargs):
        r"""Print a warning message; e.g. "foo.xml:13: blablabla"."""
        do_warn(msg_fmt, prefix=self.errprefix(), **kwargs)


    def check_token_data(self):
        r"""Check token data and emit warnings if they are malformed."""
        # We used to check if {LEMMA,FORM} contained spaces, but
        # UD 2.x now allows spaces, so we'll just skip these checks
        pass


    def iter_root_to_leaf_all_tokens(self):
        r'''Yield all Tokens in sentence, from root to leaves (aka topological sort).
        Tokens with missing HEAD information are yielded first.
        '''
        children = collections.defaultdict(list)  # dict[str, list[Token]]
        for token in self.tokens:
            if token.has_dependency_info():
                children[token['HEAD']].append(token)

        to_visit, output = collections.deque(children['0']), []
        while to_visit:
            current = to_visit.popleft()
            to_visit.extend(children[current.rank])
            output.append(current)

        seen = set(output)
        for token in self.tokens:
            if not token in seen:
                yield token
        yield from output


    def check_and_convert_categ(self, categ):
        r'''Return an updated version of given category. Warns on bad categories.'''
        if categ in Categories.KNOWN:
            return categ
        if categ in Categories.RENAMED:
            new_categ = Categories.RENAMED[categ]
            warn_once(self.errprefix(), 'Category {categ} renamed to {new_categ}',
                      categ=categ, new_categ=new_categ)
            return new_categ
        warn_once(self.errprefix(), 'Category {categ} is unknown', categ=categ)
        return categ


    def kvs_with_key(self, key: str):
        r"""Return the number of kv_pairs with the given key."""
        return sum(int(kv.key == key) for kv in self.kv_pairs)

    def print_conllup_comments(
            self, *, output=sys.stdout,
            gen_missing_req_keys=True, sent_id_key="sent_id"):
        r"""Print comments in CoNLL-UP format."""
        kv_pairs = list(self.kv_pairs)

        if gen_missing_req_keys:
            if self.kvs_with_key(sent_id_key) == 0:
                kv_pairs.insert(0, self.calc_artificial_sent_id(sent_id_key))
            if self.kvs_with_key("text") == 0:
                idx_sentid = next(i for (i, kv) in enumerate(kv_pairs) if kv.key == sent_id_key)
                kv_pairs.insert(idx_sentid+1, self.calc_artificial_text())

        for kv_pair in kv_pairs:
            print(kv_pair.to_tsv(), file=output)
        for mweid, mweoccur in enumerate(self.mweoccurs, 1):
            mweoccur.metadata.mweid = str(mweid)
            if not mweoccur.metadata.is_generically_empty():
                print(mweoccur.metadata.to_tsv(), file=output)


    def unique_kv_pair(self, key: str) -> KVPair:
        r"""Return a KVPair for given `key`.
        If the key is not unique, raises a ValueError.
        """
        ret = [kv for kv in self.kv_pairs if kv.key == key]
        if len(ret) != 1:
            raise ValueError('Metadata key is not unique: "{}" ' \
                '(found {}, expected 1)'.format(key, len(ret)))
        return ret[0]


    def keep_kv_pairs_if(self, predicate):
        r"""Keep only ToplevelComments that satisfy the predicate."""
        self.kv_pairs = [kvpair for kvpair in self.kv_pairs if predicate(kvpair)]


    def calc_artificial_text(self) -> KVPair:
        r"""Calculate required `text` attribute for CoNLL-UP."""
        return KVPair('text', ''.join(token.surface + ('' if token.nsp else ' ') for token in self.tokens))

    def calc_artificial_sent_id(self, sent_id_key="sent_id") -> KVPair:
        r"""Calculate required `sent_id` attribute for CoNLL-UP."""
        return KVPair(sent_id_key, 'autogen--{}--{}'.format(
            os.path.basename(self.corpusinfo.file_path), self.nth_sent))


class MWEOccur:
    r"""Represents an instance of a MWE in text.
    In a type/token distinction: MWELexicalItem is a type and MWEOccur is a token.

    Parameters:
    @param sentence: the sentence in which this MWEOccur was seen
    @param indexes: the indexes of the MWE inside `sentence`
    @param category: one of {VID, LVC, IRV...}
    @param metadata: an instance of MWEAnnotMetadata with extra information (may be a dummy empty instance)

    Attributes:
    @type  raw: MWEOccurView
    @param raw: represents tokens for raw form, as seen in text
    @type  fixed: MWEOccurView
    @param fixed: represents tokens in fixed form (e.g. homogenizing lemmas)
    @type  reordered: MWEOccurView
    @param reordered: represents tokens in reordered form (e.g. normalizing word-order for LVCs)
    """
    def __init__(self, sentence: Sentence, indexes: 'list[int]',
                 category: str, metadata: MWEAnnotMetadata):
        self.lang = sentence.corpusinfo.lang
        assert self.lang in LANGS
        self.sentence = sentence
        self.indexes = tuple(sorted(indexes))
        self.category = category
        self.metadata = metadata

        self.raw = MWEOccurView(self, (sentence.tokens[i] for i in indexes))
        self.fixed = self.raw._with_fixed_tokens()
        self.reordered = self.fixed._with_reordered_tokens()

        assert all(t.rank == tf.rank for (t, tf) in zip(self.raw.tokens,
                self.fixed.tokens)), "BUG: _with_fixed_tokens must preserve order"
        assert set(self.reordered.tokens) == set(self.fixed.tokens), \
                "BUG: _with_reordered_tokens must not change word attributes"

    def mweo_id(self):
        r"""Return an ID that uniquely identifies the file&sentence&indexes."""
        return (self.sentence.file_path, self.sentence.nth_sent, tuple(self.indexes))

    def __repr__(self):
        return "MWEOccur<{}>".format(" ".join(self.reordered.likely_canonicform))

    def suspiciously_similar(self, other: 'MWEOccur'):
        r'''Return True iff self and other are likely to be the same MWE
        (used e.g. to avoid predicting spurious Skipped occurrences).
        '''
        return self.sentence.file_path == other.sentence.file_path \
            and self.sentence.nth_sent == other.sentence.nth_sent \
            and (set(self.indexes) & set(other.indexes))

    def remapped_indexes(self, indexmap: 'dict[int,list[int]]'):
        r"""Remap the indexes in self based on indexmap."""
        new_indexes = [i_new  for i_old in self.indexes for i_new in indexmap.get(i_old, (i_old,))]  # flatmap
        return MWEOccur(self.sentence, new_indexes, self.category, self.metadata)

    def with_mwes_from_ranges_absorbed_into_tokens(self):
        r"""Return an MWEOccur where MWEs in ranges are moved into the element tokens."""
        if not any("-" in self.sentence.tokens[i]['ID'] for i in self.indexes):
            return self  # No ranges in this MWEOccur, nothing to do
        r2i = self.sentence.rank2index()
        indexmap = {}
        for i in self.indexes:
            wid = self.sentence.tokens[i]['ID']
            if "-" in wid:
                first, last = wid.split('-', 1)
                indexmap[i] = range(r2i(first), r2i(last)+1)
        return self.remapped_indexes(indexmap)


class MWEOccurSet:
    r'''Represents a set of MWEOccur objects.'''
    def __init__(self):
        self._mweos = collections.defaultdict(list)

    def add_mweoccur(self, mweo: MWEOccur):
        r'''Add the MWEOccur to this set.'''
        self._mweos[(mweo.sentence.file_path, mweo.sentence.nth_sent)].append(mweo)

    def add_mweoccurs_from_all(self, sentences: list):
        r'''Add MWEOccur instances for all sentences.'''
        for sent in sentences:
            for mweo in sent.mweoccurs:
                self.add_mweoccur(mweo)

    def contains_suspiciously_similar(self, mweo: MWEOccur):
        r'''Return True iff `mwe.suspiciously_similar(x)` is True for some `x` stored in `self`.'''
        for other in self._mweos[(mweo.sentence.file_path, mweo.sentence.nth_sent)]:
            if mweo.suspiciously_similar(other):
                return True
        return False


class MWEOccurView:
    r'''Represents a view of the tokens inside an MWEOccur.
    The token order may be different from the literal order in the Sentence.
    
    Parameters:
    @type  mwe_occur: MWEOccur
    @param mwe_occur: The MWEOccur that this view represents
    @type  iter_tokens: Iterable[Token]
    @param iter_tokens: Tokens for MWEs in this view (may be different from literal order in Sentence)

    Attributes:
    @type  tokens: tuple[Token]
    @param tokens: Tokens for MWEs this view (may be different from literal order in Sentence)
    @type  likely_canonicform: list[str]
    @param likely_canonicform: List of lemmas (or surfaces) for MWE tokens in this MWEOccurView.
    @type  i_head: int
    @param i_head: Index of head verb.
    @type  i_subhead: Optional[int]
    @param i_subhead: Index of subhead noun (e.g. for LVCs and some VIDs). May be `None`.
    @type  i_synroots: tuple[int]
    @param i_synroots: Index of syntactic roots (requires syntax info in CoNLL-U). Empty list if unavailable.
    '''
    def __init__(self, mwe_occur, iter_tokens):
        self.mwe_occur = mwe_occur
        self.tokens = tuple(iter_tokens)
        assert all(isinstance(t, Token) for t in self.tokens), self.tokens
        self.i_head = self._i_head()
        self.i_subhead = self._i_subhead()
        self.head = self.tokens[self.i_head]
        self.subhead = self.tokens[self.i_subhead] if (self.i_subhead is not None) else None
        self.likely_lemmatizedform = self._lemmatized_at(range(len(self.tokens)))    
        self.likely_canonicform = self._likely_canonicform()

    def _i_head(self):
        r"""Index of head verb in `likely_canonicform`
        (First word if there is no POS info available)."""
        i_verbs = [i for (i, t) in enumerate(self.tokens) if t.univ_pos == "VERB"] \
                or [(-1 if self.mwe_occur.lang in LANGS_WITH_VERB_OCCURRENCES_ON_RIGHT else 0)]
        return i_verbs[0]  # just take first verb that appears

    def _i_subhead(self):
        r"""Index of sub-head noun in `likely_canonicform` (very useful for LVCs)."""
        i_nouns = tuple(i for (i, t) in enumerate(self.tokens) if t.univ_pos == "NOUN")
        if not i_nouns: return None
        # We look for the first noun that is not the modifier in a noun compound
        head_nouns = [i for i in i_nouns if (i==len(self.tokens)-1 or self.tokens[i+1] != "NOUN")]
        return head_nouns[0]

    def _i_synroot(self):
        r"""Yield index of the syntactic roots."""
        i_nouns = tuple(i for (i, t) in enumerate(self.tokens) if t.univ_pos == "NOUN")
        if not i_nouns: return None
        # We look for the first noun that is not the modifier in a noun compound
        head_nouns = [i for i in i_nouns if (i==len(self.tokens)-1 or self.tokens[i+1] != "NOUN")]
        return head_nouns[0]

    def _i_reflpron(self):
        r"""Return the reflexive pronoun (for IRVs), or None."""
        return next((i for (i, t) in enumerate(self.tokens) if t.univ_pos == "PRON"), None)

    def _likely_canonicform(self):
        r"""Return a lemmatized form of this MWE."""
        if self.mwe_occur.lang in LANGS_WITH_ALL_CANONICAL_TOKENS_LEMATIZED:
            return self.likely_lemmatizedform
        indexes = [self.i_head, self.i_subhead, self._i_reflpron()]
        return self._lemmatized_at([i for i in indexes if i is not None])


    def _lemmatized_at(self, indexes):
        r"""Return a tuple[str] with surfaces from self.tokens, lemmatized at given indexes."""
        ret = [t.surface for t in self.tokens]
        for i in indexes:
            ret[i] = self.tokens[i].lemma_or_surface()
        return tuple(x.casefold() for x in ret)


    def _with_fixed_tokens(self):
        r"""Return a fixed version of `self.tokens` (must keep same length & order)."""
        fixed = tuple(self._fixed_token(t) for t in self.tokens)
        return MWEOccurView(self.mwe_occur, fixed)

    def _fixed_token(self, token):
        r"""Return a manually fixed version of `token` (e.g. homogenize lemmas for IRVs)."""
        if token.univ_pos == "PRON" and Categories.is_inherently_reflexive_verb(self.mwe_occur.category):
            # Normalize reflexive pronouns, e.g. FR "me" or "te" => "se"
            if self.mwe_occur.lang in ["PT", "ES", "FR"]:
                token = token.with_update(LEMMA="se")
            if self.mwe_occur.lang == "IT":
                token = token.with_update(LEMMA="si")
            if self.mwe_occur.lang == "EN":
                token = token.with_update(LEMMA="oneself")
        return token


    def _with_reordered_tokens(self):
        r"""Return a reordered version of `tokens` (must keep same length)."""
        lang, category = self.mwe_occur.lang, self.mwe_occur.category
        T, newT, iH, iS = self.tokens, list(self.tokens), self.i_head, self.i_subhead
        if Categories.is_light_verb_construction(category):
            # Reorder e.g. EN "shower take(n)" => "take shower"
            nounverb = (lang in LANGS_WITH_CANONICAL_VERB_ON_RIGHT)
            if iS is None:
                iS = 0 if nounverb else len(T)-1
            if (nounverb and iH < iS) or (not nounverb and iS < iH):
                newT[iH], newT[iS] = T[iS], T[iH]

        if Categories.is_inherently_reflexive_verb(category):
            # Reorder e.g. PT "se suicidar" => "suicidar se"
            iPron, iVerb = ((0,-1) if (lang in LANGS_WITH_CANONICAL_REFL_PRON_ON_LEFT) else (-1,0))
            if T[iVerb].univ_pos == "PRON" and T[iPron].univ_pos == "VERB":
                newT[iVerb], newT[iPron] = T[iPron], T[iVerb]
            elif lang == "PT" and (T[iVerb].univ_pos == "PART" or T[iVerb].univ_pos == "CONJ") and T[iPron].univ_pos == "VERB":
                newT[iVerb], newT[iPron] = T[iPron], T[iVerb]

        return MWEOccurView(self.mwe_occur, newT)


    def iter_root_to_leaf_mwe_tokens(self):
        r'''Yield Tokens in MWE, from closest-to-root to leaves (aka topological sort).
        Tokens with missing HEAD information are yielded first.
        '''
        # We use ranks because we sometimes replace tokens (e.g. _fixed_token above)...
        mwe_ranks = set(token.rank for token in self.tokens)
        for token in self.mwe_occur.sentence.iter_root_to_leaf_all_tokens():
            if token.rank in mwe_ranks:
                yield token


def rerooted(tokens):
    r'''Return a list of re-ranked and re-rooted tokens.
    * Tokens that referred to internal ranks new point to new ranks.
    * Tokens that referred to external ranks will now point to root:0.
    * Tokens that had parent_rank=='0' keep their old dependency info.
    '''
    ret = []
    oldrank2new = {'0': '0'}
    for t in tokens:
        oldrank2new[t.rank] = str(len(ret)+1)
        t = t.with_update(ID=oldrank2new[t.rank])

        if 'HEAD' not in t:
            pass  # leave `t` unrooted
        elif t['HEAD'] in oldrank2new:
            t = t.with_update(HEAD=oldrank2new[t['HEAD']])
        else:
            t = t.with_update(DEPREL='root', HEAD='0')
        ret.append(t)
    return ret



class RootedMWEOccur(collections.namedtuple('RootedMWEOccur', 'mweoccur rooted_tokens')):
    r"""Represents an MWEOccur along with its tokens in root-to-leaf order"""
    def n_dangling_unrooted(self):
        r"""Number of tokens that are not rooted (these are bad!)."""
        return len(self.mweoccur.raw.tokens) - len(self.rooted_tokens)

    def n_attachments_to_root(self):
        r"""Return the number of syntactic attachments where HEAD=='0'."""
        return sum(1 for t in self.rooted_tokens if t.get('HEAD') == '0')

    def cmp_key(self):
        r'''Deterministic exception-free comparison method.'''
        return [t.cmp_key() for t in self.rooted_tokens]


class RootedMWEOccurList(list):
    r"""List of RootedMWEOccur objects sharing the same lemma+syntax)."""
    def __gt__(self, other):
        r"""A RootedMWEOccurList is greater than another one if (in this order):
        * It has the fewest number of unrooted tokens (we need 0 unrooted tokens).
        * It has the fewest attachments to root (we don't like disconnected subtrees).
        * It has the most number of examples in rooted_tokens (we want the most common lemma+syntax).
        * It has smaller RootedMWEOccur.cmp_key() value (tie-breaker, for determinism).
        """
        # (all elements in a RootedMWEOccurList should have the same `n` attachments, so we just use self[0])
        if self[0].n_dangling_unrooted() < other[0].n_dangling_unrooted():
            return True
        if self[0].n_attachments_to_root() < other[0].n_attachments_to_root():
            return True
        if len(self) > len(other):
            return True
        return [t.cmp_key() for t in self] < [t.cmp_key() for t in other]



class MWELexicalItem:
    r'''Represents a group of `MWEOccur`s that share the same canonical form.
    In a type/token distinction: MWELexicalItem is a type and MWEOccur is a token.

    For example, an LVC with these MWEOccurs
      ["taking shower", "took shower", "shower taken"]
    would ideally be grouped into MWELexicalItem with canonicform "take shower".
    
    Parameters:
    @type  mweoccurs: list[MWEOccur]
    @param mweoccurs: a list of MWEOccur instances (read-only!)

    Attributes:
    @type  i_head: int
    @param i_head: index of head verb
    @type  i_subhead: Optional[int]
    @param i_subhead: index of sub-head noun
    '''
    def __init__(self, mweoccurs: list):
        self.mweoccurs = mweoccurs
        self.canonicform = most_common(m.reordered.likely_canonicform for m in mweoccurs)
        self._seen_mweoccur_ids = {m.mweo_id() for m in self.mweoccurs}  # type: set[str]

        self.i_head = most_common(m.reordered.i_head for m in mweoccurs)
        nounbased_mweos = [m.reordered.i_subhead for m in mweoccurs
                           if m.reordered.i_subhead is not None]
        self.i_subhead = most_common(nounbased_mweos, fallback=None)


    def only_non_vmwes(self):
        r'''True iff all mweoccurs are NonVMWEs.'''
        return all((o.category in Categories.NON_MWES and o.metadata.confidence is None) for o in self.mweoccurs)

    def contains_mweoccur(self, mweoccur):
        r'''True iff self.mweoccurs contains given MWEOccur.'''
        return (mweoccur.mweo_id() in self._seen_mweoccur_ids)

    def add_skipped_mweoccur(self, mweoccur):
        r'''Add MWEOccur to this MWE descriptor. If this MWEOccur already exists, does nothing.'''
        assert mweoccur.category == 'Skipped'  # we do not need to update i_head/i_subhead for Skipped
        mweoccur_id = mweoccur.mweo_id()
        if not any(mweoccur.suspiciously_similar(m) for m in self.mweoccurs):
            self._seen_mweoccur_ids.add(mweoccur_id)
            self.mweoccurs.append(mweoccur)

    def head(self):
        r'''Return a `str` with the head verb.'''
        return self.canonicform[self.i_head]

    def subhead(self):
        r'''Return a `str` with the subhead noun (fails if self.i_subhead is None).'''
        return self.canonicform[self.i_subhead]


    def most_common_rooted_tokens_and_example(self):
        r'''Return the most common output from `rerooted(mweoccur.raw.iter_root_to_leaf_mwe_tokens())`
        for all mweoccurs in `self`, along with an example MWEOccur which has these re-rooted tokens.

        @rtype RootedMWEOccur.
        '''
        lemmasyntax2rootedmweoccur = collections.defaultdict(RootedMWEOccurList)

        for mweoccur in self.mweoccurs:
            rooted_tokens = tuple(rerooted(mweoccur.raw.iter_root_to_leaf_mwe_tokens()))
            lemmasyntax = tuple((t.lemma_or_surface(), t.get('HEAD'), t.get('DEPREL')) for t in rooted_tokens)
            lemmasyntax2rootedmweoccur[lemmasyntax].append(
                RootedMWEOccur(mweoccur, rooted_tokens))  # append to RootedMWEOccurList

        majority_mweoccurs = max(lemmasyntax2rootedmweoccur.values())
        return max(majority_mweoccurs, key=RootedMWEOccur.cmp_key)


    def lemma_or_surface_list(self):
        r'''Return a list of lemmas/surfaces for elements this MWE.'''
        try:
            return self._lemma_or_surface_list
        except AttributeError:
            lsl_possibilities = [tuple(t.lemma_or_surface() for t in m.reordered.tokens)
                                 for m in self.mweoccurs]
            self._lemma_or_surface_list = most_common(lsl_possibilities)
            return self._lemma_or_surface_list



############################################################

_FALLBACK_RAISE = object()

def most_common(iterable, *, fallback=_FALLBACK_RAISE):
    r'''Utility function: Return most common element in `iterable`.
    Return `fallback` if `iterable` is empty.
    '''
    a_list = collections.Counter(iterable).most_common(1)
    if a_list:
        return a_list[0][0]

    assert fallback is not _FALLBACK_RAISE, 'Zero elements to choose from; no fallback provided'
    return fallback


class FrozenCounter(collections.Counter):
    r'''Instance of Counter that can be hashed. Should not be modified.'''
    def __hash__(self):
        return hash(frozenset(self.items()))



############################################################

# Leave the preferable PATH_FMT in PATH_FMTS[-1]
PATH_FMTS = ["{d}/{b}.conllu", "{d}/conllu/{b}.conllu"]


def calculate_conllu_paths(file_paths, warn=True):
    r"""Return CoNLL-U paths, or None on failure to find some of them."""
    ret = []
    for file_path in file_paths:
        dirname, basename = os.path.split(file_path)
        if not dirname: dirname = "."  # seriously, python...

        basename = basename_without_ext(basename)
        for path_fmt in ["{d}/{b}.conllu", "{d}/conllu/{b}.conllu"]:
            ret_path = path_fmt.format(d=dirname, b=basename)
            if os.path.exists(ret_path):
                if warn:
                    do_warn("Using CoNLL-U file `{p}`", p=ret_path, warntype="INFO")
                ret.append(ret_path)
                break

        else:
            if warn:
                do_warn("CoNLL-U file `{p}` not found", p=ret_path)
                do_warn("Not using any CoNLL-U file")
                return None
    return ret


RE_BASENAME_NOEXT = re.compile(
    r'^(?:.*/)*(.*?)(\.(folia|xml|conllu|conllup|parsemetsv|tsv|tar|gz|bz2|zip))*$')

def basename_without_ext(filepath):
    r"""Return the basename of `filepath` without any known extensions."""
    return RE_BASENAME_NOEXT.match(filepath).group(1)


#####################################################################

class IterAlignedFiles:
    r"""Class that yields Sentence instances based on file_paths and conllu_paths."""
    def __init__(
            self, lang: str, file_paths: list, conllu_paths=None,
            *, keep_nvmwes=False, default_mwe_category=None,
            keep_dup_mwes=False, keep_mwe_random_order=False, debug=False):
        self.keep_nvmwes = keep_nvmwes
        self.keep_dup_mwes = keep_dup_mwes
        self.keep_mwe_random_order = keep_mwe_random_order
        self.aligned_iterator = AlignedIterator.from_paths(
            lang, file_paths, conllu_paths, default_mwe_category=default_mwe_category, debug=debug)

    def __iter__(self):
        for sentence in self.aligned_iterator:
            assert isinstance(sentence, Sentence)
            if not self.keep_nvmwes:
                sentence.remove_non_vmwes()
            if not self.keep_dup_mwes:
                sentence.remove_duplicate_mwes()
            if not self.keep_mwe_random_order:
                sentence.mweoccurs.sort(key=lambda m: m.indexes)
            yield sentence


def _iter_parseme_file(lang, file_path, default_mwe_category):
    fileobj = open(file_path, 'r') if file_path != "-" else sys.stdin
    corpusinfo = CorpusInfo(lang, file_path, None)
    header = fileobj.buffer.peek(4096)

    if b'<FoLiA' in header:
        return FoliaIterator(corpusinfo, fileobj)
    if b'global.columns' in header:
        return ConllupIterator(corpusinfo, fileobj, default_mwe_category)

    header_lines = [x for x in header.split(b"\n") if x.strip() and not x.startswith(b"#")]
    n_cols = collections.Counter(len(x.split(b"\t")) for x in header_lines).most_common(1)[0][0]
    if n_cols == 4 : # classical parseme-tsv
        corpusinfo.colnames = "ID FORM MISC PARSEME:MWE".split()
        return ParsemeTSVIterator(corpusinfo, fileobj, default_mwe_category)
    if n_cols == 5 : # parseme-tsv with POS
        corpusinfo.colnames = "ID FORM MISC PARSEME:MWE XPOS".split()
        return ParsemeTSVIterator(corpusinfo, fileobj, default_mwe_category)
    if n_cols == 10:
        return ConlluIterator(corpusinfo, fileobj, default_mwe_category)
    if n_cols == 11: # missing header tolerated
        corpusinfo.colnames = ConlluIterator.UD_KEYS + ["PARSEME:MWE"]
        return ConllupIterator(corpusinfo, fileobj, default_mwe_category)        
    raise Exception("Unknown file format (TSV with {} columns): {!r}".format(n_cols, file_path))


class AlignedIterator:
    r"""Yield Sentence instances based on the given iterators."""
    def __init__(self, main_iterators: list, conllu_iterators: 'Optional[list]', debug=False):
        self.main_iterators = main_iterators
        self.conllu_iterators = conllu_iterators
        self.main = collections.deque(itertools.chain.from_iterable(main_iterators))
        self.conllu = collections.deque(itertools.chain.from_iterable(conllu_iterators)) if conllu_iterators else None
        self.debug = debug

    def __iter__(self):
        if self.conllu_iterators is None:
            yield from iter(self.main)
            return

        while self.main or self.conllu:
            _warn_if_none(
                self.main[0] if self.main else None,
                self.conllu[0] if self.conllu else None)
            main_s = self.main.popleft()
            conllu_s = self.conllu.popleft()

            if conllu_s:
                # Ignore all ToplevelComments in TSV (do NOT yield them)
                # Yield ToplevelComments in from CoNLL-U instead
                main_s.kv_pairs = conllu_s.kv_pairs

            tok_aligner = TokenAligner(main_s, conllu_s, debug=self.debug)
            indexmap = tok_aligner.index_mapping(main_s, conllu_s)
            main_s.re_tokenize(conllu_s.tokens, indexmap)
            yield main_s


    def merge_sentences(self, main_sentence, conllu_sentence):
        r"""Return a dict {main_i -> list[conllu_i]}"""
        tok_aligner = TokenAligner(main_sentence, conllu_sentence)
        for (info, tokens) in tok_aligner:
            if info in ["EQUAL", "CONLL:EXTRA"]:
                yield from tokens

            if self.debug and info == "CONLL:EXTRA":
                conllu_sentence.warn(
                    "Adding tokens from CoNLL: {surf!r} with rank {rank!r}",
                    surf=[t.surface for t in tokens], rank=[t.rank for t in tokens], warntype='DEBUG')


    @staticmethod
    def from_paths(lang: str, main_paths, conllu_paths, *, default_mwe_category=None, debug=False):
        r"""Return an AlignedIterator for the given paths.
        (Special case: if conllu_paths is None, return a simpler kind of iterator)."""
        chain_iter = itertools.chain.from_iterable
        main_iterators = [_iter_parseme_file(lang, p, default_mwe_category) for p in main_paths]
        conllu_iterators = None if not conllu_paths else \
            [ConlluIterator(CorpusInfo(lang, p, None), open(p, 'r'), default_mwe_category) for p in conllu_paths]
        return AlignedIterator(main_iterators, conllu_iterators, debug)


def _warn_if_none(main_sentence, conllu_sentence):
    assert conllu_sentence or main_sentence

    if not main_sentence:
        conllu_sentence.warn("CoNLL-U sentence #{n} found, but there is " \
                "no matching PARSEME-TSV input file", n=conllu_sentence.nth_sent, error=True)
    if not conllu_sentence:
        main_sentence.warn("PARSEME-TSV sentence #{n} found, but there is " \
                "no matching CoNLL-U input file", n=main_sentence.nth_sent, error=True)


class SentenceAligner:
    def __init__(self, main_sentences, conllu_sentences, debug=False):
        self.main_sentences = list(main_sentences)
        self.conllu_sentences = list(conllu_sentences)
        self.debug = debug
        main_surfs = [tuple(t.surface for t in sent.tokens) for sent in self.main_sentences]
        conllu_surfs = [tuple(t.surface for t in sent.tokens) for sent in self.conllu_sentences]
        sm = difflib.SequenceMatcher(None, main_surfs, conllu_surfs)
        self.matches_end = sm.get_matching_blocks()
        self.matches_beg = [(0, 0, 0)] + self.matches_end

    def print_mismatches(self):
        for mismatch_main, mismatch_conllu in self.mismatches():
            if mismatch_main or mismatch_conllu:
                first_conllu_sent = self.conllu_sentences[mismatch_conllu.start] \
                    if mismatch_conllu.start < len(self.conllu_sentences) else None
                first_main_sent = self.main_sentences[mismatch_main.start] \
                    if mismatch_main.start < len(self.main_sentences) else None
                _warn_if_none(first_main_sent, first_conllu_sent)

                m, c = mismatch_main.start-1, mismatch_conllu.start-1
                for m, c in zip(mismatch_main, mismatch_conllu):
                    tokalign = TokenAligner(self.main_sentences[m], self.conllu_sentences[c])
                    if not tokalign.is_alignable():
                        tokalign.msg_unalignable(error=False)
                        self.print_context(m, c)
                        do_error("Mismatched sentences")
                else:
                    # zipped ranges are OK, the error is in the end of one of the ranges
                    end_mismatch_main = range(m+1, mismatch_main.stop)
                    end_mismatch_conllu = range(c+1, mismatch_conllu.stop)
                    assert not end_mismatch_main or not end_mismatch_conllu
                    for m in end_mismatch_main:
                        self.main_sentences[m].warn(
                            "PARSEME sentence #{n} does not match anything in CoNLL-U",
                            n=self.main_sentences[m].nth_sent, error=True)
                        self.print_context(m, end_mismatch_conllu.start)
                    for c in end_mismatch_conllu:
                        self.conllu_sentences[c].warn(
                            "CoNLL-U sentence #{n} does not match anything in PARSEME",
                            n=self.conllu_sentences[c].nth_sent, error=True)
                        self.print_context(end_mismatch_main.start, c)

    def print_context(self, main_index, conllu_index):
        self.print_context_sents("PARSEME", self.main_sentences[max(0,main_index-1):main_index+2])
        self.print_context_sents("CoNLL-U", self.conllu_sentences[max(0,conllu_index-1):conllu_index+2])

    def print_context_sents(self, info, sentences):
        for sent in sentences:
            sent.warn(
                "{} sentence #{} = {} ...".format(info, sent.nth_sent,
                " ".join(t.surface for t in sent.tokens[:7])), header=False)

    def mismatches(self):
        r"""@rtype: Iterable[(mismatch_main_range, mismatch_conllu_range)]"""
        for (main1,conll1,size1), (main2,conll2,_) in zip(self.matches_beg, self.matches_end):
            yield range(main1+size1, main2), range(conll1+size1, conll2)


class TokenAligner:
    def __init__(self, main_sentence, conllu_sentence, debug=False):
        self.main_sentence = main_sentence
        self.conllu_sentence = conllu_sentence
        self.debug = debug
        main_surf = [t.surface for t in main_sentence.tokens]
        conllu_surf = [t.surface for t in conllu_sentence.tokens]
        sm = difflib.SequenceMatcher(None, main_surf, conllu_surf)
        self.matches_end = sm.get_matching_blocks()
        self.matches_beg = [(0, 0, 0)] + self.matches_end


    def index_mapping(self, main_sentence, conllu_sentence):
        r"""Return a dict {i_main -> list[i_conllu]}"""
        if not self.is_alignable():
            self.msg_unalignable(error=True)
        indexmap = collections.defaultdict(list)
        for info, range_main, range_conllu in self._triples():
            if info == "EQUAL":
                for iM, iC in zip(range_main, range_conllu):
                    indexmap[iM].append(iC)
            else:
                for iM in range_main:
                    indexmap[iM].extend(range_conllu)
                self.warn_mismatch(range_main, range_conllu)
        return indexmap


    def _triples(self):
        r"""Yield tuples (str, range_main, range_conll)."""
        # Main: [ok ok ok ok ok...ok ok ok] gap_main [ok ok ok ok...
        #       ^position=main1                      ^position=main2
        #       ^-------------size1-------^
        for (main1,conll1,size1), (main2,conll2,_) in zip(self.matches_beg, self.matches_end):
            yield ("EQUAL", range(main1, main1+size1), range(conll1, conll1+size1))
            yield ("MISMATCH", range(main1+size1, main2), range(conll1+size1, conll2))


    def is_alignable(self):
        r"""Return True iff the sentences are alignable."""
        for info, main_range, conllu_range in self._triples():
            if info == "MISMATCH" and len(main_range) > 10:
                return False
        return True


    def warn_mismatch(self, range_gap_main, range_gap_conllu):
        r"""Warn users when the two ranges do not match (one or both ranges may be empty)."""
        if range_gap_main:
            affected_mweids = [mwe_i+1 for (mwe_i, m) in enumerate(self.main_sentence.mweoccurs) \
                    if any((token_i in m.indexes) for token_i in range_gap_main)]
            self.warn_gap_main(range_gap_main, range_gap_conllu, affected_mweids)

        if range_gap_conllu:
            # Probably a range, or a sub-word inside a range
            if self.debug:
                tokens = [self.conllu_sentence.tokens[i] for i in range_gap_conllu]
                self.conllu_sentence.warn(
                        "Adding tokens from CoNLL: {surf!r} with rank {rank!r}",
                        surf=[t.surface for t in tokens], rank=[t.rank for t in tokens], warntype="DEBUG")


    def msg_unalignable(self, error=False):
        r"""Error issued if we cannot align tokens."""
        self.conllu_sentence.warn(
            "CoNLL-U sentence #{n} does not match {fileline} sentence #{n2}",
            n=self.conllu_sentence.nth_sent, fileline=self.main_sentence.errprefix(),
            n2=self.main_sentence.nth_sent, error=error)


    def warn_gap_main(self, main_range, conllu_range, all_mwe_codes):
        r"""Warn when there are unmapped characters in main file."""
        main_toks = [self.main_sentence.tokens[i].surface for i in main_range]
        #conllu_toks = [self.conllu_sentence.tokens[i].surface for i in conllu_range]

        mwe_codes_info = " (MWEs={})".format(";".join(map(str,all_mwe_codes))) if all_mwe_codes else ""
        self.main_sentence.warn(
            "Ignoring extra tokens in sentence #{n} ({fileline}): {toks!r}{mwe}",
            n=self.main_sentence.nth_sent, fileline=self.conllu_sentence.errprefix(),
            toks=main_toks, mwe=mwe_codes_info)


############################################################

class FoliaIterator:
    r"""Yield Sentence's for file_path."""
    XML_CONLLUP_SEP = "`"  # one-byte separator that is easier to read than a tab

    def __init__(self, corpusinfo, fileobj):
        self.fileobj = fileobj
        self.corpusinfo = corpusinfo
        self.nth_sent = None

    def do_warn(self, msg_fmt, **kwargs):
        prefix = os.path.basename(self.corpusinfo.file_path)
        if self.nth_sent:
            prefix += "(s.{})".format(self.nth_sent)
        do_warn(msg_fmt, prefix=prefix, **kwargs)

    def __iter__(self):
        doc = folia.Document(string=self.fileobj.read())
        doc.filename = self.fileobj
        self.colnames = None if "conllup-colnames" not in doc.metadata \
                else doc.metadata["conllup-colnames"].split(FoliaIterator.XML_CONLLUP_SEP)

        for folia_nonembedded_entitieslayer in doc.select(folia.EntitiesLayer, recursive=False):
            for folia_nonembedded_entity in folia_nonembedded_entitieslayer.select(folia.Entity):
                self.do_warn('Ignoring MWE outside the scope of a single sentence: {id!r}', id=folia_nonembedded_entity.id)

        for self.nth_sent, folia_sentence in enumerate(doc.select(folia.Sentence), 1):
            current_sentence = Sentence(self.corpusinfo, self.nth_sent, None)
            for rank, word in enumerate(folia_sentence.words(), 1):
                tokendict = {
                    'ID': str(rank),
                    'FORM': word.text() or '_',
                    'MISC': ('' if word.space else 'SpaceAfter=No'),
                }
                if self.colnames:
                    for foreign in word.select(folia.ForeignData):
                        for _xmltag, kv_elem in self.foreign_tag_elems(foreign, ["conllup-columns"]):
                            cols_str = kv_elem.attrib["columns"]
                            cols = [c.replace(FoliaIterator.XML_CONLLUP_SEP, "\t")
                                    for c in cols_str.split(FoliaIterator.XML_CONLLUP_SEP)]
                            tokendict2 = {k: v for (k, v) in zip(self.colnames, cols) if v != "_"}
                            tokendict2.pop("PARSEME:MWE", None)  # drop this info, if existent
                            tokendict.update(tokendict2)
                current_sentence.tokens.append(Token(tokendict))

            self.iter_kv_pairs(current_sentence, folia_sentence)
            folia_mwes = list(folia_sentence.select(folia.Entity))
            self.calc_mweoccurs(current_sentence, folia_mwes, folia_sentence)
            yield current_sentence


    def iter_kv_pairs(self, output_sentence: Sentence, folia_sentence):
        r"""Append instances of KVPair (or subclasses) to `output_sentence.kv_pairs`."""
        conllup_columns = []
        for fdata in folia_sentence.select((folia.ForeignData, folia.Comment), recursive=False):
            if isinstance(fdata, folia.Comment):
                output_sentence.kv_pairs.append(Metadata.from_folia(fdata))
            else:
                for _xmltag, kv_elem in self.foreign_tag_elems(fdata, ["kv-pair"]):
                    output_sentence.kv_pairs.append(KVPair(kv_elem.attrib["key"], kv_elem.attrib["value"]))


    def calc_mweoccurs(self, output_sentence: Sentence, folia_mwes, folia_sentence):
        r"""Append instances of `MWEOccur` to `output_sentence.mweoccurs`."""
        word_id2index = {w.id: i for i, w in enumerate(folia_sentence.words())}
        sent_word_ids = [w.id for w in folia_sentence.words()]
        for mwe in folia_mwes:
            mwe_word_ids = [w.id for w in mwe.wrefs()]
            if not mwe_word_ids:  # ignore empty Entities produced by FLAT
                output_sentence.warn('Ignoring empty MWE: {id!r}', id=mwe.id)
            elif any(w.id not in sent_word_ids for w in mwe.wrefs()):
                output_sentence.warn('Ignoring misplaced MWE: {id!r}', id=mwe.id)
            else:
                categ = output_sentence.check_and_convert_categ(mwe.cls)
                indexes = [word_id2index[w.id] for w in mwe.wrefs()]
                output_sentence.mweoccurs.append(MWEOccur(
                    output_sentence, indexes, categ, Metadata.from_folia(mwe)))


    def foreign_tag_elems(self, foreign: folia.ForeignData, expected_tags: list):
        r"""Yield (xmltag: str, elem: etree.Element) pairs for foreign elements
        if `tag` is in the list of `expected` tags.
        """
        for kv_elem in foreign.node.getchildren():
            xmltag = kv_elem.tag.split('}', 1)[-1]
            if xmltag in expected_tags:
                yield (xmltag, kv_elem)
            else:
                self.do_warn('Ignoring unknown foreign data in XML: <{tag}> (expecting: {exp})', tag=xmltag, exp=expected_tags)



class AbstractFileIterator:
    r'''Parameters:
    @param file_path: path to file that we will iterate (str)
    @param default_mwe_category: category to use when one is missing (str);
                                 if not specified, raises an error instead
    '''
    def __init__(self, corpusinfo, fileobj, default_mwe_category):
        self.corpusinfo = corpusinfo
        self.fileobj = fileobj
        self.default_mwe_category = default_mwe_category
        self.nth_sent = 0
        self.lineno = 0
        self._new_sent()

    def _new_sent(self):
        self.nth_sent += 1
        self.curr_sent = Sentence(self.corpusinfo, self.nth_sent, self.lineno+1)
        self.mwe_metadata = {}
        self.id2mwe_categ = {}
        self.id2mwe_ranks = collections.defaultdict(list)
        self.id2mwe_indexes = collections.defaultdict(list)

    def get_token_and_mwecodes(self, fields: list) -> (Token, list):
        r"""Return a Token and a list of mwecodes (str)
        for the list of fields in current line.
        """
        return NotImplementedError('Abstract method')

    def finish_sentence(self):
        r"""Return finished `self.curr_sent`."""
        if not self.curr_sent:
            self.warn("Unexpected empty line")
        s = self.curr_sent
        if self.default_mwe_category:
            for key in self.id2mwe_ranks:
                self.id2mwe_categ.setdefault(key, self.default_mwe_category)
        try:
            s.mweoccurs = [MWEOccur(s, tuple(self.id2mwe_indexes[id]), self.id2mwe_categ[id],
                                    self.mwe_metadata.pop(id, MWEAnnotMetadata()))
                           for id in sorted(self.id2mwe_ranks)]
            for meta in self.mwe_metadata.values():
                s.warn("Metadata with mweid {mweid!r} refers to unknown MWE", mweid=meta.mweid)
                s.kv_pairs.append(meta)
        except KeyError as e:
            self.warn("MWE has no category: {categ}", categ=e.args[0])
        self._new_sent()
        s.check_token_data()
        return s

    def warn(self, msg_fmt, **kwargs):
        do_warn(msg_fmt, prefix="{}:{}".format(self.corpusinfo.file_path, self.lineno), **kwargs)

    def make_comment(self, line):
        if self.curr_sent.tokens:
            self.warn("Comment in the middle of a sentence is not allowed")
        match = KVPair.KV_REGEX.match(line)
        keyval = KVPair.from_conllup(*match.groups()) if match else KVPair("RAWLINE", line[1:].strip())
        if isinstance(keyval, MWEAnnotMetadata):
            assert keyval.mweid not in self.mwe_metadata
            self.mwe_metadata[keyval.mweid] = keyval
        elif keyval.key == "global.columns":
            self.corpusinfo.colnames = keyval.value.strip().split()
        else:
            if keyval.key == "source_sent_id":
                keyval = KVPair("sent_id", keyval.value)
            self.curr_sent.kv_pairs.append(keyval)

    def append_token(self, line):
        token, mwecodes = self.get_token_and_mwecodes(line.split("\t"))  # method defined in subclass

        for mwecode in mwecodes:
            index_and_categ = mwecode.split(":")
            self.id2mwe_ranks[index_and_categ[0]].append(token.rank)
            self.id2mwe_indexes[index_and_categ[0]].append(len(self.curr_sent.tokens))
            if len(index_and_categ) == 2 and index_and_categ[1]:
                categ = self.curr_sent.check_and_convert_categ(index_and_categ[1])
                self.id2mwe_categ.setdefault(index_and_categ[0], categ)
        self.curr_sent.tokens.append(token)

    def __iter__(self):
        with self.fileobj:
            yield from self.iter_header(self.fileobj)
            for self.lineno, line in enumerate(self.fileobj, 1):
                try:
                    line = line.strip("\n")
                    if line.startswith("#"):
                        self.make_comment(line)
                    elif not line.strip():
                        if not self.curr_sent.empty():
                            yield self.finish_sentence()
                    else:
                        self.append_token(line)
                except:
                    self.warn("Unable to read & parse line", warntype="FATAL")
                    raise
            if not self.curr_sent.empty():
                yield self.finish_sentence()
            yield from self.iter_footer(self.fileobj)

    def iter_header(self, f):
        return []  # Nothing to yield on header

    def iter_footer(self, f):
        return []  # Nothing to yield on footer


class ConlluIterator(AbstractFileIterator):
    UD_KEYS = 'ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC'.split()
    def __init__(self, corpusinfo, fileobj, default_mwe_category):
        corpusinfo.colnames = self.UD_KEYS
        super().__init__(corpusinfo, fileobj, default_mwe_category)

    def get_token_and_mwecodes(self, data):
        if len(data) != 10:
            self.warn("CoNLL-U line has {n} columns, not 10", n=len(data))
        return Token(zip(self.UD_KEYS, data)), []


class ConllupIterator(AbstractFileIterator):
    def get_token_and_mwecodes(self, data):
        if len(data) != len(self.corpusinfo.colnames):
            self.warn("Line has {n} columns, not {n_exp}", n=len(data), n_exp=len(self.corpusinfo.colnames))
        tokendict = dict(zip(self.corpusinfo.colnames, data))
        mwe_codes = tokendict.pop('PARSEME:MWE') or "_"
        m = mwe_codes.split(";") if mwe_codes not in "_*" else []
        return Token(tokendict), m


class ParsemeTSVIterator(AbstractFileIterator):
    def __init__(self, corpusinfo, fileobj, default_mwe_category):        
        super().__init__(corpusinfo, fileobj, default_mwe_category)

    def get_token_and_mwecodes(self, data):
        if len(data) == 5:
            xpos = data[4]
        else:
            xpos = EMPTY
            #warn_once(
            #    "{}:{}".format(self.corpusinfo.file_path, self.lineno),
            #    "Silently ignoring 5th parsemetsv column")
            #data.pop()  # remove data[-1]
        if len(data) != 4:
            self.warn("PARSEMETSV line has {n} columns, not 4", n=len(data))
        # ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC
        conllu = {
            'ID': data[0],
            'FORM': data[1] or '_',
            'MISC': ('SpaceAfter=No' if data[2]=='nsp' else ''),
            'XPOS': xpos,
        }
        mwe_codes = data[3]
        m = mwe_codes.split(";") if mwe_codes not in "_*" else []
        return Token(conllu), m



############################################################

class AbstractWriter:
    def __init__(self, *, output=sys.stdout):
        self.output = output
        self.sentno = 0

    def write_sentences(self, sents: list):
        r"""Write all sentences to output file."""
        for sent in sents:
            self.sentno += 1
            self._do_write_sentence(sent)

    def _do_write_sentence(self, sent: Sentence):
        r"""(Must be defined in subclasses)."""
        raise NotImplementedError


class ConllupWriter(AbstractWriter):
    def __init__(self, *, output=sys.stdout, colnames=None, keepranges=False):
        super().__init__(output=output)
        self.colnames = colnames
        self.keepranges = keepranges

    def _do_write_sentence(self, sent):
        if self.sentno == 1:
            self.colnames = self.colnames or sent.corpusinfo.colnames \
                or ConlluIterator.UD_KEYS + ["PARSEME:MWE"]
            print("# global.columns =", " ".join(self.colnames), file=self.output)
            sent.keep_kv_pairs_if(lambda x: x.key != 'global.columns')

        sent.print_conllup_comments(output=self.output)
        for token, mwecodes in sent.tokens_and_mwecodes():
            line = "\t".join(self.token_values(token, self.colnames, mwecodes))
            print(line, file=self.output)
        print(file=self.output)

    def token_values(self, token: Token, colnames: list, mwecodes: 'list[str]'):
        for colname in colnames:
            if colname == 'PARSEME:MWE':
                if "-" in token["ID"] and not self.keepranges:
                    yield "*" # The PARSEME:MWE column is not defined for multiword tokens <= CHANGE THIS IN THE FUTURE PUT _
                else:
                    yield ";".join(mwecodes) or "*"
            else:
                yield token.get(colname, EMPTY)


############################################################

_WARNED = set()

def warn_once(first_seen_here, msg_fmt, **kwargs):
    r"""Same as do_warn, but only called once per msg_fmt."""
    actual_msg = msg_fmt.format(**kwargs)
    if actual_msg not in _WARNED:
        _WARNED.add(actual_msg)
        do_warn(msg_fmt, **kwargs)
        do_warn('First seen here: {here}', here=first_seen_here, header=False)
        do_warn('(Ignoring further warnings of this type)', header=False)


def do_info(msg_fmt, **kwargs):
    r"""Same as do_warn, but using warntype=="INFO"."""
    do_warn(msg_fmt, warntype="INFO", **kwargs)

def do_error(msg_fmt, **kwargs):
    r"""Same as do_warn, but using warntype=="ERROR" (calls exit)."""
    do_warn(msg_fmt, warntype="ERROR", **kwargs)


def do_warn(msg_fmt, *, prefix=None, warntype=None, error=False, header=True, **kwargs):
    r"""Print a warning message "prefix: msg"; e.g. "foo.xml:13: blablabla"."""
    warntype = warntype or ("ERROR" if error else "WARNING")
    dot_warntype = warntype if header else '.'*len(warntype)

    msg = msg_fmt.format(**kwargs)
    prefix = "{}: {}".format(prefix, dot_warntype) if prefix else dot_warntype
    final_msg = "{}: {}".format(prefix, msg)

    if COLOR_STDERR:
        if not header:
            color = '38;5;245'  # ANSI color: grey
        elif warntype == "ERROR":
            color = 31          # ANSI color: red
        elif warntype == "FATAL":
            color = '7;31'      # ANSI color: red+invert
        elif warntype == "INFO":
            color = 34          # ANSI color: blue
        else:
            color = 33  # ANSI color: yellow
        final_msg = "\x1b[{}m{}\x1b[m".format(color, final_msg)

    print(final_msg, file=sys.stderr)
    if warntype == "ERROR":
        exit(1)


############################################################

class InputContext:
    r"""Context manager to indicate sentence where an error occurred."""
    def __init__(self, sent: Sentence, token: Token = None):
        self.sent = sent
        self.token = token

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_value and not isinstance(exc_value, SystemExit):
            if self.token:
                self.sent.warn('Error when processing token {tokid} in this sentence',
                    tokid=self.token.get('ID', '??'), warntype="FATAL")
            else:
                self.sent.warn('Error when processing this sentence', warntype="FATAL")


############################################################

def read_mwelexitems(iter_sentences):
    r"""Return two lists: (list[MWELexicalItem], list[MWELexicalItem]).
    The first list concerns real MWEs, while the second concerns strictly NonVMWEs.
    """
    lf2mweoccurs = _lemmatizedform2mweoccurs(iter_sentences)  # type: dict[tuple[str], list[MWEOccur]]
    lemmatizedform2mwe_mixed = collections.OrderedDict()  # type: dict[tuple[str], MWELexicalItem]
    lemmatizedform2mwe_nvmwe = collections.OrderedDict()  # type: dict[tuple[str], MWELexicalItem]

    for lemmatizedform, mweoccurs in lf2mweoccurs.items():
        mwe = MWELexicalItem(mweoccurs)
        if mwe.only_non_vmwes():
            lemmatizedform2mwe_nvmwe[lemmatizedform] = mwe
        else:
            lemmatizedform2mwe_mixed[lemmatizedform] = mwe
    return (list(lemmatizedform2mwe_mixed.values()), list(lemmatizedform2mwe_nvmwe.values()))


def _lemmatizedform2mweoccurs(iter_sentences):
    r'''Return a dict[tuple[str], list[MWEOccur]].'''
    ret = collections.defaultdict(list)  # type: dict[tuple[str], list[MWEOccur]]
    for sentence in iter_sentences:
        for mwe_occur in sentence.mweoccurs:
            lemmatizedform = FrozenCounter(mwe_occur.reordered.likely_lemmatizedform)
            ret[lemmatizedform].append(mwe_occur)
    return ret


############################################################

SKIPPED_FINDER_PATTERNS = '{Dependency, UnlabeledDep, BagOfDeps, WindowGapX}'


def skipped_finder(finding_method, lang, mwes, *, favor_precision):
    r'''Return an instance of a subclass of AbstractSkippedFinder.

    Parameters:
    @type  finding_method: str
    @param finding_method: A string matching one of the SKIPPED_FINDER_PATTERNS
    @type  lang: str
    @param lang: one of the languages from the `LANGS` global
    @type  mwes: list[MWELexicalItem]
    @param mwes: list of MWEs to be used for annotation (it may be ignored)
    @type  favor_precision: bool
    @param favor_precision: whether to favor precision (conservative) or recall
    '''
    if finding_method == 'Dependency':
        return DependencyBasedSkippedFinder(lang, mwes, favor_precision, 'LABELED-ARC')
    if finding_method == 'UnlabeledDep':
        return DependencyBasedSkippedFinder(lang, mwes, favor_precision, 'UNLABELED-ARC')
    if finding_method == 'BagOfDeps':
        return DependencyBasedSkippedFinder(lang, mwes, favor_precision, 'BAG')

    if finding_method.startswith('WindowGap'):
        max_gaps = int(finding_method[len('WindowGap'):])
        return WindowBasedSkippedFinder(lang, mwes, favor_precision, max_gaps)

    raise ValueError('Invalid finding-method: `{}`'.format(finding_method))



class AbstractSkippedFinder:
    r'''Superclass for algorithms to find skipped MWEs in text.

    Parameters:
    @type  lang: str
    @param lang: one of the languages from the `LANGS` global
    @type  mwes: list[MWELexicalItem]
    @param mwes: list of MWEs to be used for annotation
    @type  favor_precision: bool
    @param favor_precision: whether to favor precision (conservative) or recall
    '''
    def __init__(self, lang, mwes, favor_precision):
        self.lang = lang
        self.mwes = list(mwes)
        self.favor_precision = favor_precision

    def find_skipped_in(self, sentences):
        r"""Yield pairs (MWELexicalItem, MWEOccur) for Skipped MWEs in all sentences."""
        raise NotImplementedError

    def _mweinfo_pair(self, mwelexitem, sentence, indexes):
        r"""Return a pair (MWELexicalItem, MWEOccur) with
        a new MWEOccur for "Skipped" MWE in sentence (at given indexes).
        """
        indexes = tuple(sorted(indexes))
        metadata = MWEAnnotMetadata(annotator='autodetect', annotatortype='auto')
        new_occur = MWEOccur(sentence, indexes, "Skipped", metadata)
        return (mwelexitem, new_occur)




class WindowBasedSkippedFinder(AbstractSkippedFinder):
    r'''Algorithm to find skipped MWEs based on a window
    (allows gaps of up to `max_gaps` words).
    See documentation in AbstractSkippedFinder.
    
    Extra Parameters:
    @type  max_gaps: int
    @param max_gaps: max number of gaps between words in match
    '''
    def __init__(self, lang, mwes, favor_precision, max_gaps):
        super().__init__(lang, mwes, favor_precision)
        self.max_gaps = max_gaps

        self.mweelement2mwes = collections.defaultdict(list)  # type: dict[str, list[MWELexicalItem]]
        for mwe in self.mwes:
            for lemmasurface in set(mwe.lemma_or_surface_list()):
                self.mweelement2mwes[lemmasurface.lower()].append(mwe)

    def find_skipped_in(self, sentences):
        r"""Yield pairs (MWELexicalItem, MWEOccur) for Skipped MWEs in all sentences."""
        for sentence in sentences:
            for i, token in enumerate(sentence.tokens):
                for wordform in [token.lemma_or_surface().lower(), token.surface.lower()]:
                    for mwe in self.mweelement2mwes.get(wordform, []):
                        yield from self._find_skipped_mwe_at(sentence, mwe, i)

    def _find_skipped_mwe_at(self, sentence, mwe, i_head):
        r"""Yield a Skipped MWE or nothing at all."""
        unmatched_words = collections.Counter(
            x.lower() for x in mwe.lemma_or_surface_list())
        matched_indexes = []

        def matched(wordform, i):
            unmatched_words[wordform] -= 1
            if unmatched_words[wordform] == 0:
                unmatched_words.pop(wordform)  # so verbose...
            matched_indexes.append(i)

        gaps = 0
        for i in range(i_head, len(sentence.tokens)):
            word = sentence.tokens[i]
            if not unmatched_words:
                break  # matched_indexes is complete
            if gaps > self.max_gaps:
                break  # failed to match under max_gaps
            if word.surface.lower() in unmatched_words:
                matched(word.surface.lower(), i)
            elif word.lemma_or_surface().lower() in unmatched_words:
                matched(word.lemma.lower(), i)
            else:
                gaps += 1

        if not unmatched_words:
            yield self._mweinfo_pair(mwe, sentence, matched_indexes)



#------------------------------------------------------------

class Bag:
    r'''Bag[A, B] is similar to a Dict[A, set[B]], with insertions adding to the set of values.'''
    def __init__(self, key_value_pairs=()):
        self.dict = {}
        for k, value in key_value_pairs:
            try:
                subset = self.dict[k]
            except KeyError:
                subset = self.dict[k] = set()
            subset.add(value)

    def is_empty(self):
        return (not self.dict)

    def __contains__(self, key):
        return (key in self.dict)

    def __getitem__(self, key):
        return self.dict.get(key, frozenset())

    def excluding(self, key, value_in_bag):
        r'''Return new Bag excluding (key, value_in_bag).'''
        return Bag((k, v) for (k, values) in self.dict.items() for v in values if (k, v) != (key, value_in_bag))


class MWEBagFrame(collections.namedtuple('MWEBagFrame', 'mwe n_roots lemmabag')):
    r'''Attributes:
    @type  mwe: MWELexicalItem
    @type  n_roots: int
    @type  n_roots: Number of tokens with parent '0'
    @type  lemmabag: Bag[str, Token]
    @param lemmabag: Mapping from lemma to set of rooted tokens
    '''

class DependencyBasedSkippedFinder(AbstractSkippedFinder):
    r'''Algorithm to find skipped MWEs based
    on bags of unlabeled syntactic dependencies.
    See documentation in AbstractSkippedFinder.

    Extra parameters:
    @type  matchability: str
    @param matchability: One of {labeled-arc, unlabeled-arc, bag}
    '''
    def __init__(self, lang, mwes, favor_precision, matchability):
        super().__init__(lang, mwes, favor_precision)
        self.matchability = matchability
        self.rootmostlemma2mwebagframe = collections.defaultdict(list)  # type: dict[str, list[MWEBagFrame]]

        for mwe in self.mwes:
            # CAREFUL: re-rooting changes the parent_rank of dependencies
            rootedmweoccur = mwe.most_common_rooted_tokens_and_example()
            example_mweoccur, rooted_tokens = rootedmweoccur
            n_roots = rootedmweoccur.n_attachments_to_root()

            assert rootedmweoccur.n_dangling_unrooted() == 0, rootedmweoccur
            if not all(t.has_dependency_info() for t in rooted_tokens):
                example_mweoccur.sentence.warn(
                    'Skipping MWE with partial dependency info: {}'.format("_".join(mwe.canonicform)))
                continue

            if favor_precision and n_roots > 1:
                example_mweoccur.sentence.warn(
                    'Skipping MWE with disconnected syntax tree: {}'.format('_'.join(mwe.canonicform)))
                continue

            x = MWEBagFrame(mwe, n_roots, Bag((t.lemma_or_surface().lower(), t) for t in rooted_tokens))
            self.rootmostlemma2mwebagframe[mwe.head().lower()].append(x)


    def find_skipped_in(self, sentences):
        r"""Yield pairs (MWELexicalItem, MWEOccur) for Skipped MWEs in all sentences."""
        for sentence in sentences:
            reordered_sentence_tokens = tuple(sentence.iter_root_to_leaf_all_tokens())

            # For every rootmost lemma in sentence, find all MWEOccurs involving this lemma
            for rootmost_lemma in sorted(set(t.lemma_or_surface().lower() for t in reordered_sentence_tokens)):
                for mwebagframe in self.rootmostlemma2mwebagframe.get(rootmost_lemma, []):
                    sub_finder = _SingleMWEFinder(
                            self.lang, self.favor_precision, self.matchability, sentence,
                            reordered_sentence_tokens, mwebagframe.mwe, mwebagframe.n_roots, mwebagframe.lemmabag)

                    for matched_indexes in sub_finder.find_indexes():
                        yield self._mweinfo_pair(mwebagframe.mwe, sentence, matched_indexes)


class _SingleMWEFinder(collections.namedtuple(
        '_SingleMWEFinder',
        'lang favor_precision matchability sentence reordered_sentence_tokens mwe max_roots lemmabag')):
    r'''Finder of all occurrences of `mwe` in `reordered_sentence_tokens`.'''

    def find_indexes(self):
        r"""Yield Skipped MWE occurrence indexes in sentence (may yield 2+ MWEs in rare cases)."""
        assert len(self.mwe.canonicform) == sum(len(v) for v in self.lemmabag.dict.values()), \
                (self.mwe.canonicform, self.lemmabag.dict)
        rank2index = self.sentence.rank2index()

        already_matched = MWEBagAlreadyMatched.EMPTY
        for matched_ranks in self._recursive_find_ranks(0, already_matched, self.lemmabag):
            assert len(matched_ranks) == len(self.mwe.canonicform), self.mwe.canonicform
            yield tuple(rank2index[rank] for rank in matched_ranks)


    def _recursive_find_ranks(self, i_start, already_matched, unmatched_lemmabag):
        r'''Yield sets of ranks fully matching current MWE.'''
        if unmatched_lemmabag.is_empty():
            yield already_matched.rank2rootedrank.keys()
            return

        for i, sentence_token, rooted_token in self._find_matched_tokens(
                 i_start, already_matched, unmatched_lemmabag):
            new_already_matched = already_matched.including(sentence_token, rooted_token)
            new_unmatched_lemmabag = unmatched_lemmabag.excluding(rooted_token.lemma_or_surface().lower(), rooted_token)
            yield from self._recursive_find_ranks(i+1, new_already_matched, new_unmatched_lemmabag)


    def _find_matched_tokens(self, i_start, already_matched, unmatched_lemmabag):
        r'''Yield all (i, sentence_token, rooted_token) for matches at reordered_sentence_tokens[i].'''
        for i, sentence_token in enumerate(self.reordered_sentence_tokens[i_start:], i_start):
            if not sentence_token.has_dependency_info():
                continue  # If we have no dependency info, avoid false positives

            for wordform in [sentence_token.lemma_or_surface().lower(), sentence_token.surface.lower()]:
                for rooted_token in unmatched_lemmabag[wordform]:
                    match_triple = (i, sentence_token, rooted_token)

                    if sentence_token['HEAD'] in already_matched.rank2rootedrank:
                        # Non-rootmost token, connected to someone in `already_matched`
                        expected_rooted_parent_rank = already_matched.rank2rootedrank[sentence_token['HEAD']]
                        if self._matches_in_tree(i, sentence_token, rooted_token, already_matched, expected_rooted_parent_rank):
                            yield match_triple

                    elif already_matched.n_roots < self.max_roots:
                        # Rootmost token, does not have a parent inside `already_matched`
                        if self._matches_rootmost(i, sentence_token, rooted_token, already_matched):
                            yield match_triple


    def _matches_rootmost(self, i, sentence_token, rooted_token, already_matched):
        r'''Return True iff `sentence_token` matches a rootmost element of self.mwe'''
        if self.matchability == 'BAG':
            return True
        if self.matchability in ('LABELED-ARC', 'UNLABELED-ARC'):
            # Only allow if it was expected to attach to root
            return rooted_token['HEAD'] == '0'
        assert False, self.matchability


    def _matches_in_tree(self, i, sentence_token, rooted_token, already_matched, expected_rooted_parent_rank):
        r'''Return True iff `sentence_token` matches non-rootmost element of self.mwe'''
        if self.matchability == 'BAG':
            return True
        if self.matchability == 'UNLABELED-ARC':
            return expected_rooted_parent_rank == rooted_token['HEAD']
        if self.matchability == 'LABELED-ARC':
            return (expected_rooted_parent_rank == rooted_token['HEAD']
                    and sentence_token['DEPREL'] == rooted_token['DEPREL'])
        assert False, self.matchability


class MWEBagAlreadyMatched(collections.namedtuple('MWEBagAlreadyMatched', 'rank2rootedrank n_roots')):
    r'''Attributes:
    @type  rank2rootedrank: dict[str,str]
    @param rank2rootedrank: Mapping from rank in sentence to rank in rooted_tokens
    @type  n_roots: int
    @param n_roots: Number of roots already matched (greater than 1 for disconnected trees)
    '''
    def including(self, sentence_token, rooted_token):
        assert sentence_token.rank not in self.rank2rootedrank, \
                ('Already matched!', sentence_token, self.rank2rootedrank)
        new_rank2rootedrank = dict(self.rank2rootedrank)
        new_rank2rootedrank[sentence_token.rank] = rooted_token.rank
        new_n_roots = self.n_roots + int(sentence_token.get('HEAD', '0') not in self.rank2rootedrank)
        return MWEBagAlreadyMatched(new_rank2rootedrank, new_n_roots)

MWEBagAlreadyMatched.EMPTY = MWEBagAlreadyMatched({}, 0)
