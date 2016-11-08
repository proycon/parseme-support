TSV-to-FoLiA conversion tool
================================

Installation
---------------

- Clone this repository: ``git clone https://github.com/proycon/parseme-support`` (requires a git client, see https://help.github.com/articles/set-up-git/)
- Install Python 3 if not yet available on your system.
    - For Debian/Ubuntu Linux: ``sudo apt-get install python3 python3-pip python3-distutils``.
    - Windows/Mac users can use the Anaconda distribution: https://www.continuum.io/downloads
- Run ``python3 setup.py install``
    - For global installation on Debian/Ubuntu Linux: ``sudo python3 setup.py install``
    - For Anaconda or virtual environments: ``python3 setup.py install``

Usage 
--------

- Run the script as follows, from the directory where you cloned this
  repository: ``tsv2folia /path/to/your/file.tsv`` (multiple file arguments allowed) 
- The output will be a similarly named file, but with extension ``folia.xml``.
  This can be uploaded to FLAT.

Input Formats
--------------

Input files should be in one of the PARSEME TSV formats, as described [here](http://typo.uni-konstanz.de/parseme/index.php/2-general/142-parseme-shared-task-on-automatic-detection-of-verbal-mwes#format).

