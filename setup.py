
from setuptools import find_packages, setup
import io

with open('VERSION') as fd:
    VERSION = fd.read().strip()

setup(
    name='simpleh5',
    version=VERSION,
    description='hdf5 table column store built on pytables',
    long_description=io.open('README.md', 'r', encoding='utf-8').read(),
    classifiers=[''],
    keywords='',
    author='Soren Solari',
    author_email='sorensolari@gmail.com',
    url='',
    license='MIT License',
    packages=find_packages(exclude=['tests']),
    package_data={},
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'blosc',
        'msgpack>=1.0.0',
        'numpy',
        'tables',
    ],
    entry_points={
    },
)