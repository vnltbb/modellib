from setuptools import setup, find_packages

setup(
    name='modellib',
    version='0.1.0',
    description='A custom Python module for streamlining plant disease detection model development pipeline.',
    author='vnltbb',  
    author_email='myhajung@gmail.com',
    url='https://github.com/vnltbb',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'timm',
        'numpy',
        'scikit-learn>=1.1',
        'matplotlib',
        'seaborn',
        'torchinfo',
        'grad-cam',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)