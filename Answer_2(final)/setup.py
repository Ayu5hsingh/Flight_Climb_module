from setuptools import setup, find_packages

setup(
    name="ml_model_sita_internship",
    version="0.1.0",
    description="A simple machine learning package for classification and regression",
    author="Ayush Singh", 
    author_email="ayush.singh@hec.edu",  
    packages=find_packages(), 
    install_requires=[
        "numpy>=1.19.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.3.0",
        "plotly>=5.0.0",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)