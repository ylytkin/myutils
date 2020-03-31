from setuptools import setup

requirements = [
    "networkx~=2.4",
    "nltk~=3.5",
    "numpy~=1.19.1",
    "pymystem3~=0.2.0",
    "scipy~=1.5.2",
]

setup(
    name='myutils',
    version='0.2',
    description='Some small Python utility functions I frequently use.',
    url='https://github.com/ylytkin/myutils',
    author='Yura Lytkin',
    author_email='jurasicus@gmail.com',
    license='MIT',
    packages=['myutils'],
    install_requires=requirements,
    zip_safe=False,
)
