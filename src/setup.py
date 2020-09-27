from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['tensorflow>=2.3',
'cloudml-hypertune']


setup(
    name='models',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='my first csustom package with dependencies.')
