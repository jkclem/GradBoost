# -*- coding: utf-8 -*-
"""
Created on Saturday 20:35:29 2020
@author: jkcle
"""

from setuptools import setup

setup(name='GradBoost',
      version=' 1.0',
      description='Package to perform gradient boosting using sklearn regression models.',
      url='https://github.com/jkclem/GradBoost',
      author='John Clements',
      author_email='jkclements2016@gmail.com',
      license='MIT',
      packages=['chowtest'],
      install_requires=[
          'numpy', 'typing', 'tqdm'
      ],
      zip_safe=False)
