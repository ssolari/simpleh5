
from setuptools import find_packages, setup
import io


setup(
    name='pyh5column',
    version='0.0.1',
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
        'pytables',
    ],
    entry_points={
    },
)