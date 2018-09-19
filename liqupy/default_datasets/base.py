# **************************************************************************************
# LiquPy: Open-source Python Library for Soil Liquefaction and Lateral Spread Analysis
# https://github.com/LiquPy/LiquPy
# **************************************************************************************


from os.path import dirname, join
from pandas import read_excel



def load_youd_bartlett_demo():
    """
    load and return a demo version of the database compiled by Youd et al. (2002)
    This dataset is a classic and very common dataset.
    Most of the lateral spread empirical models are build on top of this dataset.
    """
    return read_excel(join(dirname(__file__),'YoudHansenBartlett2002_demo.xls'))

def load_spt_idriss_boulanger():
    """
    """
    return read_excel(join(dirname(__file__),'spt_Idriss_Boulanger.xlsx'))
