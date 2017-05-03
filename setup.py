from setuptools import setup
from sydar.version import __version__
def readme():
    with open('README.rst') as f:
        return f.read()

def version():   
    try:
        with open('VERSION') as f:
            return f.read() 
    except:
        pass
    try:
        with open('sydar/versi') as f:
            return f.read() 
    except:
        pass

setup(name='sydar',
      version=__version__,
      description='Synthesis Done Approximately Right',
      long_description=readme(),
      url='https://github.com/u-t-autonomous/sydar.git',
      author='Mohammed Alshiekh',
      author_email='sahabi@gmail.com',
      license='BSD',
      packages=['sydar'],
      install_requires=[
          'pyparsing',
          'numpy',
          ],
      entry_points = {
        'console_scripts': ['sydar-matlab=sydar.command_line:main'],
        },
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)
