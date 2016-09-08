TSV-to-FoLiA conversion tool
================================

Installation
---------------

- Clone this repository: ``git clone https://github.com/proycon/parseme-support`` (requires a git client, see https://help.github.com/articles/set-up-git/)
- Install Python 3 if not yet available on your system.
    - For Debian/Ubuntu Linux: ``sudo apt-get install python3 python3-pip``.
    - Windows/Mac users can use the Anaconda distribution: https://www.continuum.io/downloads
- Install dependency *pynlpl*: 
    - ``sudo pip3 install pynlpl`` (if you want to install it globally on Debian/Ubuntu)
    - ``pip install pynlpl`` (if you use Anaconda)

Usage 
--------

- Run the script as follows, from the directory where you cloned this
  repository: ``python3 tsv2folia.py [tsvfile] [[tsvfile]]`` 

