#! /usr/bin/env python3

r"""
This module defines MWE category labels and helper functions.
Everything should be part of the class Categories, which is imported by `dataalign.py`.

Avoid using this module directly (use `dataalign.Categories` instead).
"""


class Categories:
    r'''This class contains the code referring to MWE categories.
    See the `KNOWN` constant for the set of MWE categories.
    '''

    # List of all categories in `Category` class
    KNOWN = {
        'VID',
        'LVC.full',
        'LVC.cause',
        'IRV',
        'VPC.full',
        'VPC.semi',
        'MVC',
        'IAV',
        'NotMWE',
        'TODO',
    }

    # Subset of KNOWN, with the categories that represent non-MWEs
    NON_MWES = set(['NotMWE', 'TODO'])

    # Mapping of categories from ST 1.0 to ST 1.1
    RENAMED = {
        'ID': 'VID',
        'OTH': 'TODO',
        'IReflV': 'IRV',
        'LVC': 'LVC.full',
        'VPC': 'VPC.full',
        'NonVMWE': 'NotMWE',
    }

    assert NON_MWES.issubset(KNOWN)
    assert set(RENAMED.values()).issubset(KNOWN)


    @staticmethod
    def is_light_verb_construction(str_category):
        r'''True iff `str_category` is an LVC'''
        return str_category.startswith('LVC.')

    @staticmethod
    def is_inherently_reflexive_verb(str_category):
        r'''True iff `str_category` is an IRV'''
        return str_category == 'IRV'


    @staticmethod
    def css_for_labels():
        r'''Get CSS for category labels'''
        return '''
                .mwe-label-NotMWE { background-color: #DCC8C8; }
                .mwe-label-Skipped { background-color: #DDDDDD; }
                .mwe-label-TODO { background-color: #AA0000; }                
                
                .mwe-label-VID { background-color: #FF6AFF; }
                .mwe-label-LVC-full { background-color: #9AA6FF; }
                .mwe-label-LVC-cause { background-color: #9AA6FF; }
                .mwe-label-VPC-full { background-color: #CC8833; }
                .mwe-label-VPC-semi { background-color: #CC8833; }
                .mwe-label-IRV { background-color: #FFB138; }
                .mwe-label-MVC { background-color: #C13AC1; }
                .mwe-label-IAV { background-color: #AAAAAA; }
            '''

    @staticmethod
    def css_name(str_category_label):
        r'''Get CSS name for given category label.'''
        from xml.sax.saxutils import escape as ESC
        return 'mwe-label-{}'.format(ESC(str_category_label.replace('.', '-')))

    @staticmethod
    def consistency_check_mwe_pairs():
        r'''Yield (category, annot_info) pairs.'''
        yield  ('VID',       'Annotate as VID (idiom)')
        yield  ('LVC.full',  'Annotate as LVC.full (light-verb)')
        yield  ('LVC.cause', 'Annotate as LVC.cause (light-verb)')
        yield  ('IRV',       'Annotate as IRV (reflexive)')
        yield  ('VPC.full',  'Annotate as VPC.full (verb-particle)')
        yield  ('VPC.semi',  'Annotate as VPC.semi (verb-particle)')
        yield  ('MVC',       'Annotate as MVC (multi-verb)')
        yield  ('IAV',       'Annotate as IAV (adpositional)')

    @staticmethod
    def consistency_check_nonmwe_pairs():
        r'''Yield (category, annot_info) pairs.'''
        yield  ('NotMWE', 'Mark as not-an-MWE')
