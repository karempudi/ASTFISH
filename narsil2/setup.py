from setuptools import setup, find_packages

setup(
     name='narsil2',
        version='0.0.1',
        author='Praneeth Karempudi',
        author_email='praneeth.karempudi@gmail.com',
        license='LICENSE.txt',
        description='The package to work with microscopy data of prokaryotic cells',
        long_description=open('README.md').read(),
        url="https://github.com/karempudi/narsil2",
        install_requires=[
            'numpy',
            'torch',
            'torchvision',
            'scikit-image',
            'scipy',
            'scikit-learn',
            'seaborn',
            'numba',
            'fastremap',
            'ncolor',
            'edt',
            'tqdm',
            'pytorch-lightning',
            'opencv-python',
            'matplotlib',
            #'wxPython',
            'pyqtgraph',
            ],
        packages=find_packages(exclude=("tests", "notebooks", "data", "saved_models"))
        )