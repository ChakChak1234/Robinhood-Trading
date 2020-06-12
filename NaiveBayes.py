import pandas as pd
import numpy as np
from utilities import *

class NaiveBayes:
    def __init__(self,data):
        self.columns = list(data.columns)
    
    