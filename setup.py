# **************************************************************************************
# LiquPy: Open-source Python Library for Soil Liquefaction and Lateral Spread Analysis
# https://github.com/LiquPy/LiquPy
# **************************************************************************************

from setuptools import setup, find_packages

##with open("Readme.md", "r") as fh:
##    long_description = fh.read()
    
setup(name='LiquPy',
      version='0.13.1',
      description='Open-source Python Library for Soil Liquefaction and Lateral Spread Analysis',
      url='https://github.com/LiquPy/LiquPy',
      author='Massoud Hosseinali',
      author_email='massoud.hosseinali@utah.edu',
      license='New BSD License',
      long_description='Open-source Python Library for Soil Liquefaction and Lateral Spread Analysis',
      packages = find_packages(), 
      install_requires = [
          'numpy',
          'matplotlib',
          'pandas',
          'scikit-learn',
          'openpyxl'],
      zip_safe=False)


