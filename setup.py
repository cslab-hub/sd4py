from distutils.core import setup

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='SD4Py',
    version='0.1.4',
    author='Dan Hudson',
    author_email='daniel.dominic.hudson@uni-osnabrueck.de',
    packages=['sd4py'],
    license='Licence.txt',
    description='Subgroup discovery and visualisation for python based on the VIKAMINE kernel',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        "License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)",
        "Programming Language :: Python :: 3"
    ],
    install_requires=[
        "JPype1",
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "networkx"
    ],
    include_package_data=True
)
