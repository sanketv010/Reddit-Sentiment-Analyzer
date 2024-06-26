from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path:str)->list:
    requirements = []
    with open(file_path, 'r') as file_obj:
        requirements = file_obj.readlines()
        [req.replace("\n", "") for req in requirements]

        if "-e ." in requirements:
            requirements.remove("-e .")

    return requirements        

setup(
    name='MLOps Project',
    version='0.0.1',
    author='sanket',
    author_email='sanketverma425@gmail.com',
    packages=find_packages(include=['src', 'src.*']),
    install_requires=get_requirements('requirements.txt'),
)