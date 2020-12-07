from setuptools import setup

with open('requirements.txt') as file:
    requirements = [line.strip() for line in file.readlines()]

setup(
    name='myutils',
    version='0.3',
    description='Some small Python utility functions I frequently use.',
    url='https://github.com/ylytkin/myutils',
    author='Yura Lytkin',
    author_email='jurasicus@gmail.com',
    license='MIT',
    packages=['myutils'],
    install_requires=requirements,
    zip_safe=False,
)
