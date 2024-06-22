"""Setup package"""

import os
from setuptools import setup, find_packages

# Packages to include in the distribution
packages = find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"])
exclude_from_cythonize = ["entities"]

# Additional data required to install this package
package_data = {}

# Files with that are data out of the package
# data_files=[('my_data', ['data/data_file'])],

# List of dependencies minimally needed by this project to run
with open('requirements.in') as f:
    install_requires = [x for x in f.read().splitlines() if not x.startswith(('--', '#'))]

# Trove classifiers
classifiers = [
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Developers',
    'Natural Language :: English',
    'Topic :: Software Development :: Tools',
    'License :: Non OSI Approved :: Copyright',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: Implementation :: CPython',
    'Programming Language :: Python :: Implementation :: PyPy'
]

# Keywords to help users find this package on PyPi
keywords = ''

here = os.path.abspath(os.path.dirname(__file__))
meta = {}
readme = ''

# Read version and README files
with open(os.path.join(here, 'gebold', '_meta.py'), 'r') as f:
    exec(f.read(), meta)
with open(os.path.join(here, 'README.md'), 'r') as f:
    readme = f.read()

setup(
    name=meta['__title__'],
    version=meta['__version__'],
    description=meta['__description__'],
    long_description=readme,
    long_description_content_type='text/markdown',
    url=meta['__url__'],
    author=meta['__author__'],
    author_email=meta['__email__'],
    license=meta['__license__'],
    classifiers=classifiers,
    keywords=keywords,
    platforms=['any'],
    packages=packages,
    install_requires=install_requires,
    package_data=package_data,
    include_package_data=True,
    python_requires=">=3, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*, !=3.5.*, !=3.6.*, !=3.7.*, <4",
    entry_points=f"""
        [console_scripts]
        {meta['__title__']}={meta['__title__']}.cli:main
    """,
)