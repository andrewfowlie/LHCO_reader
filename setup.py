from distutils.core import setup

setup(name='LHCO_reader',
      packages=['LHCO_reader'],  # this must be the same as the name above
      version='1.6.0',
      install_requires=['prettytable', 'numpy'],
      package_data={'example': ['example.lhco']},
      description='Module for reading and analyzing LHCO files',
      author='Andrew Fowlie',
      author_email='Andrew.Fowlie@Monash.Edu.Au',
      url='https://github.com/innisfree/LHCO_reader',
      download_url='https://github.com/innisfree/LHCO_reader/tarball/1.6.0',
      keywords=['LHCO', 'HEP', 'ROOT'],
      classifiers=['Development Status :: 5 - Production/Stable',
                   'Programming Language :: Python :: 2.7',
                   ],
      )
