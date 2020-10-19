'''
Automatically creates documentation, runs unit tests, and checks pep8
Assumes that package to run is the same name as the folder within which this file exists
'''

import glob
import os
import re
import shutil
import subprocess

PEP8FILE = 'pep8_compliance'
SOURCE = 'docs'
DOCS = 'docs/html'


# CONFIGURATION
IGNORE = []
# should be False and rerun when pushing docs to git.  set to true to create local HTML docs for debuging.
CREATE_LOCAL_DOCS = True

# get directory name
REPO_NAME = os.path.basename(os.path.dirname(os.path.abspath(__file__)))


def _modify_file(filename, line_callback):
    """
    Perform individual line modifications through the call back function to modify a file

    :param filename:
    :param line_callback:
    :return:
    """
    tmp_file = "%s.tmp" % filename
    outf = open(tmp_file, 'w')
    with open(filename) as f:
        for i, line in enumerate(f):
            line = line_callback(i, line)
            if line == "**Exit**":
                break
            elif line is not None:
                outf.write(line)
    outf.close()
    os.remove(filename)
    os.rename(tmp_file, filename)

# get name of directory that includes this file
main_dir = os.path.basename(os.path.dirname(os.path.abspath(__file__)))

# remove all .pyc files
os.system('find %s/ -name "*.pyc" -delete' % main_dir)

try:
    os.remove(PEP8FILE)
except:
    pass
# run pep8 compliance testing
os.system('pep8 --max-line-length=120 %s/ > %s' % (main_dir, PEP8FILE))
if os.path.exists(PEP8FILE) and os.path.getsize(PEP8FILE) == 0:
    os.remove(PEP8FILE)

try:
    shutil.rmtree(SOURCE)
except:
    print("Didn't remove %s" % SOURCE)
    pass

# run apidoc
max_depth = 2
cmd = f'sphinx-apidoc -F -e -o {SOURCE} {main_dir}'
result = subprocess.run(cmd.split(' '))


# modify conf.py file to make modifications
def _mod_conf(i, line):
    if re.match('\#\s*sys.path', line):
        line = "import sys\nimport os\nsys.path.insert(0, os.path.abspath('..'))\n"
    elif re.match('\#add_module_names =', line):
        line = 'add_module_names = False'
    elif re.match('html_theme', line):
        line = '# ' + line
    elif re.match('extensions = ', line):
        line = line.strip() + " 'sphinx_autodoc_annotation', \n"
        # line = line.strip() + " 'sphinx_autodoc_annotation', 'sphinxcontrib.napoleon', \n"
    elif re.search('sphinx-quickstart on', line):
        line = None
    elif re.search('^master_doc =', line):
        line = line + "\nautodoc_member_order = 'bysource'\n"

    return line

_modify_file('%s/conf.py' % SOURCE, _mod_conf)


# modify rst files
def _mod_rst(i, line):
    line = line.rstrip()
    line = re.sub(".*\.(\w+) (module|package)$", "\g<1> \g<2>", line)

    for pattern in IGNORE:
        if re.search(re.escape(pattern), line):
            return None
    # # include subclasses
    if re.search(":members:", line):
        newline = re.sub("members", "inherited-members", line)
        line = line + "\n" + newline
    if re.search("tests", line):
        # do not print tests
        line = None
    elif re.search("^(Subpackages|Submodules|---)", line):
        line = None
    elif re.search("Module contents", line):
        line = "**Exit**"
    elif re.search('modindex', line):
        line = None
    elif re.search('sphinx-quickstart on', line):
        line = None
    else:
        line += "\n"
    return line


# remove long paths to every module/package for a cleaner look
for filename in glob.glob("%s/*.rst" % SOURCE):

    # check if rst is a package to add in code for __init__.py processing
    is_package = False
    tmp_file = "%s.tmp" % filename
    outf = open(tmp_file, 'w')
    with open(filename) as f:
        for i, line in enumerate(f):
            if i == 0 and re.search('package$', line):
                is_package = True
            elif i == 2 and is_package:
                pack_name = os.path.basename(filename).replace('.rst', '')
                line = f'\n.. automodule:: {pack_name}\n    :members:\n'
            outf.write(line)
    outf.close()
    os.remove(filename)
    os.rename(tmp_file, filename)

    # perform basic .rst modifications
    _modify_file(filename, _mod_rst)
    if re.search(".*\.test.*", filename):
        os.remove(filename)
    else:
        for pattern in IGNORE:
            if re.search(re.escape(pattern), filename):
                os.remove(filename)

# make new index.rst
os.rename(f'{SOURCE}/{REPO_NAME}.rst', f'{SOURCE}/index.rst')

# call build
if CREATE_LOCAL_DOCS:
    cmd = f'sphinx-build -b html {SOURCE} {DOCS}'
    result = subprocess.run(cmd.split(' '))
