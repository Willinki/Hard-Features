from setuptools import setup, find_packages

setup(
    name="src",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "hydra-core>=1.3",
        "torch>=2.2",
        "torchvision>=0.22",
        "pytorch-lightning>=2.5",
        "hydra_submitit_launcher>=1.1.0",
    ],
    extras_require={
        "log": ["wandb>=0.19"],
    },
    python_requires=">=3.11, <4",
    author="Davide Badalotti",
    author_email="davide.badalotti@unibocconi.it",
    description="Simple project",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
