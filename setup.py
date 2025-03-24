from setuptools import setup, find_packages


setup(
    name="flexygen",
    version="0.0.3",
    packages=find_packages("src"),
    package_dir={"": "src"},
    author="Yuang Cai",
    description="Flexible generation interface for HF generative models.",
    url="https://github.com/Liadrinz/flexygen"
)
