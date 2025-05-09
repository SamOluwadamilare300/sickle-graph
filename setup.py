from setuptools import setup, find_packages

setup(
    name="sicklegraph",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests",
        # Add other dependencies as needed
    ],
    entry_points={
        'console_scripts': [
            'sicklegraph=sicklegraph.cli:main',
        ],
    },
)