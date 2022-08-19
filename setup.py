import setuptools
from os import path
import surfreact

here = path.abspath(path.dirname(__file__))
AUTHORS = """
Brenden Pelkie
Stephanie Valleau
"""


# Get the long description from the README file
with open(path.join(here, 'README.md')) as f:
    long_description = f.read()

if __name__ == "__main__":
    setuptools.setup(
        name='surfreact',
        version='0.1',
        author=AUTHORS,
        project_urls={
            'Source': 'https://github.com/valleau-lab/surface_reactions',
        },
        description=
        'tools for ML with surface reactions and CFF',
        long_description=long_description,
        include_package_data=False, #no data yet, True if we want to include data
        keywords=[
            'Machine Learning', 'Reaction Kinetics', 'Ab initio',
            'Chemical Engineering','Chemistry', 
        ],
        license='Apache License 2.0',
        packages=setuptools.find_packages(exclude="tests"),
        scripts = [], #if we want to include shell scripts we make in the install
        install_requires=[
            'numpy', 
            'pandas', 
            'ase',
	    'seaborn',
            'cathub',
            'sqlalchemy',
            'matplotlib',
            'scipy'
        ],
        extras_require={
            'tests': [
                'pytest',
                'coverage',
                'flake8',
                'flake8-docstrings'
            ],
            'docs': [
                'sphinx',
                'sphinx_rtd_theme',

            ]
        },
        classifiers=[
            'Development Status :: 1 - Planning',
            'Environment :: Console',
            'Operating System :: OS Independant',
            'Programming Language :: Python',
            'Topic :: Scientific/Engineering',
        ],
        zip_safe=False,
    )
