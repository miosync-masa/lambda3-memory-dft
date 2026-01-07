"""
Memory-DFT:Direct Schrödinger Evolution (DSE)
"""

from setuptools import setup, find_packages

setup(
    name="memory_dft",
    version="0.1.0",
    author="Masamichi Iizumi, Tamaki Iizumi",
    author_email="",
    description="Direct Schrödinger Evolution (DSE)",
    long_description=open("README.md").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "gpu": ["cupy>=10.0.0"],
        "jax": ["jax", "jaxlib"],
        "full": ["cupy>=10.0.0", "jax", "jaxlib"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
