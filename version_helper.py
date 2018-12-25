"""This helper lets different parts of the code and the setup.py access the 
value of version specified in the VERSION file.

TODO
----
Use something like `packaging` if possible so as to not reinvent wheel.
"""

import os

def get_version(base_dir=os.path.dirname(__file__)):
    """Returns both the long version stored in the VERSION file as well as the 
    short version calculated from it.
    """
    with open(os.path.join(base_dir, 'VERSION')) as version_file:
        long_v = version_file.read().strip()
        short_v = '.'.join(long_v.split('.')[0:-1])
        return long_v, short_v