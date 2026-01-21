from setuptools import setup, find_packages

setup(
    name="fibercert",
    version="0.1.0",
    description="A Provably Secure V2X Behavior Authentication Framework",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "torch>=1.10.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
        "pyyaml>=5.4.0",
    ],
    python_requires=">=3.8",
)

