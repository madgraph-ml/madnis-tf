from setuptools import setup, find_packages
import platform

HTTPS_GITHUB_URL = "https://github.com/ramonpeter/MadNIS"
PROCESSOR = platform.processor()

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = ['scikit-hep', "pandas", "scipy", "tables"]
if 'arm' in PROCESSOR.lower():
    requirements.insert(0, "tensorflow-macos")
else:
    requirements.append("tensorflow-gpu")

setup(
    name="madnis-tf",
    version="1.0.1",
    author="Ramon Winterhalder, Theo Heimel",
    author_email="ramon.winterhalder@uclouvain.be",
    description="Machine learning for neural multi-channel importance sampling in MadGraph",
    long_description=long_description,
    long_description_content_type="text/rst",
    url=HTTPS_GITHUB_URL,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    packages=find_packages(),
    install_requires=requirements,
)
