from setuptools import find_packages, setup

setup(
    setup_requires=['pbr'],
    pbr=True,
    packages=find_packages(include=['statstests']),
)