from setuptools import setup, find_packages

setup(
    name="confEMD",
    version="0.1.0",
    description="TBD",
    author="TBD",
    author_email="TBD",
    packages=find_packages(),
    install_requires=[
       'colorednoise>=2.2.0',
       'emd>=0.7.0',
       'numpy>=1.26.4',
       'lmfit>=1.3.2'
    ],
    python_requires=">=3.6",
)