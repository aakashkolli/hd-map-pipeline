from setuptools import find_packages, setup


setup(
    name="hd-map-pipeline",
    version="0.1.0",
    package_dir={"": "."},
    packages=find_packages(include=["src", "src.*", "configs"]),
    python_requires=">=3.10",
)

