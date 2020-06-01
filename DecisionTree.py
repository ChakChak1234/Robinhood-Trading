import numpy as np
import pandas as pd
import math
import random

class Node:
    def __init__(self,feature=None,threshold=0,gini=0):
        self.left = None # left child
        self.right = None # right child
        self.is_terminal = False
        self.feature = feature # feature node represents
        self.threshold = threshold # threshold value for node
        self.gini = gini # split cost

    def print(self):
        print('Feature:',self.feature)
        print('Threshold:',self.threshold)
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
            print('%s[%s < %.3f] Gini = %.4f' % 
                (depth*' ',node.feature, node.threshold, node.gini))
            self.print(node.left, depth+1)
            self.print(node.right, depth+1)
        else:
            print('%s[%s]' % (depth*' ',node))            

    def partition(self,data,feature,value):
        true_df = {}
        false_df = {}
        tickers = list(data.index)
        
        for row in tickers:
            if data.loc[row,feature] < value:
                true_df[row] = list(data.loc[row,:].values)
            else:
                false_df[row] = list(data.loc[row,:].values)
        
        true_df = pd.DataFrame.from_dict(true_df,orient='index',columns=data.columns)
        false_df = pd.DataFrame.from_dict(false_df,orient='index',columns=data.columns)

        #print(true_df,false_df)
        return true_df, false_df
    
    def gini_index(self,groups):
        gini = 0.0
        instances = float(sum([len(group) for group in groups]))
        
        for group in groups:
            n = float(len(group))
            if n == 0:
                continue
            
            t = sum(group['Actual Signal'] == True)/n
            f = sum(group['Actual Signal'] == False)/n
            gini += (1-(math.pow(t,2) + math.pow(f,2))) * (n/instances)

        gini = round(gini,4)
        #print(gini)
        return gini

    # returns node that results in the optimal split
    def best_split(self,data):
        best_feature,best_threshold,best_gini,best_groups = None,math.inf,math.inf,None
        cols = list(data.columns)
        tickers = list(data.index)
        
        for column in cols[:-1]:
            for row in tickers:
                groups = self.partition(data,column,data.loc[row,column])
                gini = self.gini_index(groups)
                
                #print('%s < %.4f Gini= %.4f' % (column,data.loc[row,column],gini))
                if gini < best_gini:
                    best_feature = column
                    best_threshold = data.loc[row,column]
                    best_gini = gini
                    best_groups = groups
        
        node = Node(best_feature,best_threshold,best_gini)
        node.left,node.right = best_groups
        #node.print()
        return node
    
    # make a node terminal
    def terminal(self,group):
        vals = group['Actual Signal']
        t = sum(vals == True)
        f = sum(vals == False)
        if t >= f:
            return True
        else:
            return False

    # generate left and right children for a node or make it terminal
    def split(self,node,depth,min_size):
        left,right = node.left,node.right
        
        # check for no split
        if left is None or right is None:
            node.left,node.right = self.terminal(left+right)
            return node
        
        # check for max depth
        if depth >= self.max_depth:
            node.left = self.terminal(left)
            node.right = self.terminal(right)
            return node
        
        # process left child
        if len(left) <= min_size:
            node.left = self.terminal(left)
        else:
            node.left = self.best_split(left)
            self.split(node.left, depth+1,min_size)
        
        # process right child
        if len(right) <= min_size:
            node.right = self.terminal(right)
        else:
            node.right = self.best_split(right)
            self.split(node.right, depth+1,min_size)
        
    # build the decision tree
    def build_tree(self,data,min_size):
        root = self.best_split(data)
        self.split(root,1,min_size)
        return root

    # predict based on a row of data
    def predict(self,node,data):
        if data[node.feature] < node.threshold:
            if isinstance(node.left,Node):
                return self.predict(node.left,data)
            else:
                return node.left
        else:
            if isinstance(node.right,Node):
                return self.predict(node.right,data)
            else:
                return node.right

if __name__ == '__main__':
    df = pd.read_csv('decision-tree-data/test1.csv',index_col='Ticker')
    #df.drop('Golden Cross',axis=1,inplace=True)
    
    model = DecisionTree()
    tree = model.build_tree(df,1)
    model.print(tree)