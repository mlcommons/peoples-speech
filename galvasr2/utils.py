import os
import re
import sys

# https://stackoverflow.com/a/45176191
def find_runfiles():
    """Find the runfiles tree (useful when _not_ run from a zip file)"""
    # Follow symlinks, looking for my module space
    stub_filename = os.path.abspath(sys.argv[0])
    while True:
        # Found it?
        module_space = stub_filename + '.runfiles'
        if os.path.isdir(module_space):
            break

        runfiles_pattern = r"(.*\.runfiles)"
        matchobj = re.match(runfiles_pattern, os.path.abspath(sys.argv[0]))
        if matchobj:
            module_space = matchobj.group(1)
            break

        raise RuntimeError('Cannot find .runfiles directory for %s' %
                           sys.argv[0])
    return module_space
