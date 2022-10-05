from setuptools import setup, find_packages

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content]

setup(
    name="DefautClientStreamlit",
    version="1.0",
    description="DefautClient streamlit",
    packages=find_packages(),
    include_package_data=True,  # includes in package files from MANIFEST.in
    install_requires=requirements)
