from setuptools import setup, find_packages

setup(
    name="gcbh2",
    version="0.0.1",
    packages=find_packages(),
    scripts=[
        "./gcbh2/viz/process_bh_run.py"
    ],
)
