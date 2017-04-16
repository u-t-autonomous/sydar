from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

def version():
    with open('sydar/version.txt') as f:
        v= f.read()
        return v

setup(name='sydar',
      version=version(),
      description='Synthesis Done Approximately Right',
      long_description=readme(),
      url='https://github.com/u-t-autonomous/sydar.git',
      author='Mohammed Alshiekh',
      author_email='sahabi@gmail.com',
      license='BSD',
      packages=['sydar'],
      install_requires=[
          'scipy',
          'pyparsing',
          'numpy',
          ],
      entry_points = {
        'console_scripts': ['sydar-matlab=sydar.command_line:main'],
        },
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)