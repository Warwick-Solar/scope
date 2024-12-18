from setuptools import setup, find_packages

setup(
    name="scope",
    version="0.1.0",
    description="SCOPE is the project to detect quasi-periodicities in the \
        solar atmosphere using the EMD technique. These oscillatory signals \
        are typically accompanied by a combination of white noise and coloured \
        noise with power law spectral dependence. To detect quasi-periodicities, \
        we compute the EMD spectrum containing EMD-revealed modes and the \
        confidence limits of modal energy. This allows us to identify the \
        significant mode beyond the confidence limits, which is expected to be \
        associated with the quasi-periodic oscillatory signal of interest.",
    author="Dmitrii Kolotkov, Weijie Gu, Sergey Belov, Valery Nakariakov",
    author_email="Sergey.Belov@warwick.ac.uk",
    packages=find_packages(),
    install_requires=[
       'colorednoise>=2.2.0',
       'emd>=0.7.0',
       'numpy>=1.26.4',
       'lmfit>=1.3.2'
    ],
    python_requires=">=3.6",
)