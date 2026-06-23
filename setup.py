from setuptools import setup, find_packages

setup(
    name="llm4mof",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "openai",
        "pandas",
        "python-dotenv",
        "requests",
        "scipy",
    ],
    extras_require={
        "raspa": [
            "raspa3 @ conda-forge",  # Install via: conda install raspa3 -c conda-forge
        ],
    },
    python_requires=">=3.9",
)
