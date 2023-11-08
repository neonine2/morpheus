from setuptools import find_packages, setup


# read version file
exec(open('src/version.py').read())

# extras_require = {
#     'tensorflow': ['tensorflow>=2.0.0, !=2.6.0, !=2.6.1, <2.11.0'],
#     'torch': ['torch>=1.9.0, <2.0.0'],
#     'all': [
#         'tensorflow>=2.0.0, !=2.6.0, !=2.6.1, <2.11.0',
#         'torch>=1.9.0, <2.0.0'
#     ]
# }

if __name__ == '__main__':
    setup(name='morpheus',
          version=__version__,  # type: ignore # noqa F821
          author='Zitong Jerry Wang',
          author_email='jerry.wang95@yahoo.ca',
          description='Morpheus is an intergrated deep learning framework for generating therapeutic strategies using counterfactual optimization',
          long_description=open('README.md').read(),
          long_description_content_type='text/markdown',
          url='https://github.com/neonine2/morpheus',
          license="Apache 2.0",
          packages=find_packages(),
          include_package_data=True,
          python_requires='>=3.8, <4',
          # lower bounds based on Debian Stable versions where available
          install_requires=[
              'numpy>=1.21.6, <2.0.0',
              'pandas>=1.3.5, <2.0.0',
              'scipy>=1.7.3, <2.0.0',
              'torch>=1.13.1, <2.0.0',
              'tensorflow>=2.11.0, <3.0.0',
              'pytorch-lightning>=2.1.1, <3.0.0',
            #   'scikit-learn>=1.0.0, <2.0.0',
            #   'spacy[lookups]>=2.0.0, <4.0.0',
            #   'blis<0.8.0',  # Windows memory issues https://github.com/explosion/thinc/issues/771
            #   'scikit-image>=0.17.2, <0.20',  # introduced `start_label` argument for `slic`
            #   'requests>=2.21.0, <3.0.0',
            #   'attrs>=19.2.0, <23.0.0',
              'matplotlib>=3.0.0, <4.0.0',
              'dill>=0.3.0, <0.4.0',
              'transformers>=4.7.0, <5.0.0',
              'tqdm>=4.28.1, <5.0.0',
          ],
          test_suite='tests',
          zip_safe=False,
          classifiers=[
              "Intended Audience :: Science/Research",
              "Operating System :: OS Independent",
              "Programming Language :: Python :: 3",
              "Programming Language :: Python :: 3.7",
              "Programming Language :: Python :: 3.8",
              "Programming Language :: Python :: 3.9",
              "Programming Language :: Python :: 3.10",
              "License :: OSI Approved :: Apache Software License",
              "Topic :: Scientific/Engineering",
          ])
