from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='sydar',
      version='0.13',
      description='Synthesis Done Approximately Right',
      long_description=readme(),
      url='https://github.com/u-t-autonomous/sydar.git',
      author='Mohammed Alshiekh',
      author_email='sahabi@gmail.com',
      license='BSD',
      packages=['sydar'],
      entry_points = {
        'console_scripts': ['sydar-matlab=sydar.command_line:main'],
        },
      zip_safe=False)