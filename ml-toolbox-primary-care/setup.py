"""
Setup configuration for ML Toolbox - Railway Edition
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ml-toolbox-primary-care",
    version="0.1.0",
    author="Your Name",
    description="ML Toolbox for Virtual Primary Care - Railway Optimized",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.0.3",
        "numpy>=1.24.3",
        "scikit-learn>=1.3.0",
        "fastapi>=0.104.1",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.4.2",
        "pydantic-settings>=2.0.3",
        "click>=8.1.7",
        "joblib>=1.3.2",
        "redis>=5.0.1",
        "python-dotenv>=1.0.0",
        "python-multipart>=0.0.6",
    ],
    entry_points={
        "console_scripts": [
            "ml-toolbox=ml_toolbox.cli:cli",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)