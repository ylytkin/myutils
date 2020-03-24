from setuptools import setup, find_packages

requirements = """networkx>=2.4
numpy>=1.18
scipy>=1.4""".splitlines()

setup(
    name='utils',
    version='0.1',
    description='Some small Python utility functions I frequently use.',
    url='https://github.com/ylytkin/utils',
    author='Yura Lytkin',
    author_email='jurasicus@gmail.com',
    license='MIT',
    packages=find_packages(),
    install_requires=requirements,
    zip_safe=False,
)
