from setuptools import find_packages, setup
import pathlib
import pkg_resources

with pathlib.Path('requirements.txt').open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement
        in pkg_resources.parse_requirements(requirements_txt)
    ]

setup(
    name='heart_disease',
    packages=find_packages(),
    version='0.1.0',
    description='Predicting heart disease',
    author='Evgenii Polikutin',
    license='MIT',
    install_requires=install_requires
)
