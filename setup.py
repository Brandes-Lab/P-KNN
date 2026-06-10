from setuptools import setup, find_packages

setup(
    name="P_KNN",
    version="1.0.1",
    description="P-KNN command line tool",
    author="Po-Yu Lin and Nadav Brandes",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24",
        "pandas>=1.5",
        "scikit-learn>=1.2",
        "tqdm>=4.60",
        "huggingface_hub>=0.16"
    ],
    extras_require={
        "gpu": ["torch>=2.0"],
        "cpu": ["joblib>=1.2"],
        "all": ["torch>=2.0", "joblib>=1.2"]
    },
    entry_points={
        "console_scripts": [
            "P_KNN = P_KNN.P_KNN:main",
            "P_KNN_config = P_KNN.P_KNN_config:main",
            "P_KNN_memory_estimator = P_KNN.P_KNN_memory_estimator:main"
        ]
    },
    package_data={
        "P_KNN": ["*.py"]
    },
    include_package_data=True
)