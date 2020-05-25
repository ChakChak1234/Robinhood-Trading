import numpy as np
import pandas as pd
import math
import random

class Node:
    def __init__(self):
        self.left = None # left child
        self.right = None # right child
        self.is_terminal = False
        self.feat = None # feature node represents
        self.thresh = 0 # threshold value for node
        self.gini = 0 # split cost

    def print(self):
        print('Feature:',self.feat)
        print('Threshold:',self.thresh)
        print('Terminal?:',self.is_terminal)
        print('Cost:',self.gini)
        print('Left:\n',self.left)
        print('Right:\n',self.right)
        

class DecisionTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth

    # prints a visual representation of the tree
    def print(self,node,depth=0):
        if isinstance(node,Node):
            print('%s[%s < %.3f] Gini = %.4f' % (depth*' ',node.feat, node.thresh, node.cost))
            self.print(node.left, depth+1)
            self.print(node.right, depth+1)
        else:
            print('%s[%s]' % (depth*' ',node))

    def check_question(self,data,row,feature,value):
        val = data.loc[row,feature]
        return val >= value

    def partition(self,data,question):
        true_df = {}
        false_df = {}
        tickers = list(data.index)
        
        for t in tickers:
            if self.check_question(data,t,):
                true_df[t] = list(data.loc[t,:].values)
            else:
                false_df[t] = list(data.loc[t,:].values)
        
        true_df = pd.DataFrame.from_dict(true_df,orient='index',columns=data.columns)
        false_df = pd.DataFrame.from_dict(false_df,orient='index',columns=data.columns)

        return true_df, false_df
    
    def gini_helper(self,data):
        tickers = list(data.index)
        counts = {}
        for t in tickers:
            label = data.loc[t,'Actual Signal']
            if label not in counts:
                counts[label] = 0
            counts[label] += 1
        return counts
            
    def gini_index(self,data):
        impurity = 1
        counts = self.gini_helper(data)
        print(counts)
        
        for label in counts:
            prob = counts[label] / float(len(data))
            impurity -= prob ** 2
            
        print(impurity)
        return round(impurity,4)
        

    # returns node that results in the optimal split
    def best_split(self,data):
        pass
            
    # make a node terminal
    def terminal(self,group):
        pass

    # generate left and right children for a node or make it terminal
    def split(self, node, max_depth, min_size, depth):
        pass

    # build the decision tree
    def build_tree(self,train, max_depth, min_size=1):
        pass

    # predict based on a row of data
    def predict(self,node,data):
        pass

if __name__ == '__main__':
    df = pd.read_csv('dt_train.csv')
    
    