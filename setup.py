from setuptools import setup, find_packages

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="nlp-transformer",
    version="0.1.0",
    description="A PyTorch-based Transformer model with Grouped Query Attention and KV caching",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Shaan Vats",
    author_email="shaanvats3121@gmail.com",
    url="https://github.com/S-VATS31/nlp-transformer",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
    ],
    extras_require={
        "flash": ["flash-attn>=2.0.0;platform_system!='Windows'"],
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=5.0.0",
            "mypy>=0.990",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="transformer pytorch attention gqa kv-cache deep-learning nlp machine-learning",
)
