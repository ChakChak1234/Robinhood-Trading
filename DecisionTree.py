import numpy as np
import pandas as pd
import math
import random

class Node:
    def __init__(self):
        self.left = None # left child
        self.right = None # right child
        self.feat = None # feature node represents
        self.thresh = 0 # threshold value for node
        self.cost = 0 # split cost

    def print(self):
        print('Feature:',self.feat)
        print('Threshold:',self.thresh)
        print('Cost:',self.cost)
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

    def cost_function(self,groups):
        inst = float(sum([len(group) for group in groups]))
        cost = 0.0
        for group in groups:
            size = len(group)
            if size == 0:
                continue
            y = sum(group['Indicator'] == True)/size
            n = sum(group['Indicator'] == False)/size
            cost += (1-(math.pow(y,2) + math.pow(n,2))) * (size/inst)
        cost = round(cost,5)
        return cost

    # splits data into groups
    def best_split_helper(self,col,val,data):
        n = len(data)
        left,right = [],[]
        for row in data.index:
            if data.loc[row,col] < val:
                left.append(row)
            else:
                right.append(row)

        left = data.loc[left,]
        right = data.loc[right,]
        return left,right

    # returns node that results in the optimal split
    def best_split(self,data):
        n = len(data)
        best_feat = None
        best_thresh = math.inf
        best_cost = math.inf
        best_groups = None

        cols = list(data.columns)
        #cols.pop()
        for col in cols[:-1]:
            #print(col)
            for row in data.index:
                groups = self.best_split_helper(col,data.loc[row,col],data)
                cost = self.cost_function(groups)
                #print(cost)

                if cost <= best_cost:
                    best_feat = col
                    best_thresh = data.loc[row,col]
                    best_cost = cost
                    best_groups = groups

        # return node that represents optimal split
        node = Node()
        node.feat = best_feat
        node.thresh = best_thresh
        node.cost = best_cost
        node.left,node.right = best_groups
        return node
            
    # make a node terminal
    def terminal(self,group):
        vals = group['14day Profit']
        t = sum(vals == True)
        f = sum(vals == False)
        if t >= f:
            return True
        else:
            return False

    # generate left and right children for a node or make it terminal
    def split(self, node, max_depth, min_size, depth):
        left,right = node.left,node.right
        #print(left,right)

        # check for no split
        if left is None or right is None:
            node.left,node.right = self.terminal(left + right)
            return

        # check if tree has reached the maximum depth
        if depth >= max_depth:
            node.left = self.terminal(left)
            node.right = self.terminal(right)
            return

        # left child
        if len(left) <= min_size:
            node.left = self.terminal(left)
        else:
            node.left = self.best_split(left)
            self.split(node.left, self.max_depth, min_size, depth+1)

        # right child
        if len(right) <= min_size:
            node.right = self.terminal(right)
        else:
            node.right = self.best_split(right)
            self.split(node.right, self.max_depth, min_size, depth+1)

    # build the decision tree
    def build_tree(self,train, max_depth, min_size=1):
        root = self.best_split(train) 
        self.split(root, self.max_depth, min_size, 1)
        return root

    # predict based on a row of data
    def predict(self,node,data):
        if data[node.feat] < node.thresh:
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
    df = pd.read_csv('dt_train.csv')
    df.set_index('Ticker',inplace=True)
    
    test1 = pd.read_csv('dt_test1.csv')
    test2 = pd.read_csv('dt_test2.csv')

    tree = DecisionTree()
    node = tree.best_split(df)
    node.print()
    print(list(df.columns))
    #root = tree.build_tree(df,9,1)
    #tree.print(root,1)
    '''
    preds = []
    for t in test1.index:
        pred = tree.predict(root,test1.loc[t,])
        preds.append(pred)
    '''