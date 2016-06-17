import os

from setuptools import setup
from pip.req import parse_requirements

requirements_txt = os.path.join(os.path.dirname(__file__), 'requirements.txt')
install_reqs = parse_requirements(requirements_txt, session=False)
reqs = [str(ir.req) for ir in install_reqs]

def readme():
    README_rst = os.path.join(os.path.dirname(__file__), 'README.rst')
    with open(README_rst) as f:
        return f.read()


setup(name='yapic_io',
      version='0.0.0',
      description='io data handling module for various image sources as interface for pixel classification tools',
      long_description=readme(),
      #url='http://github.com/storborg/funniest',
      author='Manuel Schoelling, Christoph Moehl',
      author_email='manuel.schoelling@dzne.de, christoph.moehl@dzne.de',
      packages=['yapic_io'],
      zip_safe=False,
      install_requires=reqs,
      test_suite='nose.collector',
      tests_require=['nose', 'coverage', 'nose-timer'])

