#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 10:16:51 2020

@author: tungbioinfo
"""

import itertools

class FPNode(object):
    
    def __init__(self, value, count, parent):
        """
        Create the node

        Parameters
        ----------
        value : TYPE
            DESCRIPTION.
        count : TYPE
            DESCRIPTION.
        parent : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.value = value 
        self.count = count
        self.parent = parent
        self.link = None
        self.children = []
        
    def has_child(self, value):
        
        for node in self.children:
            if node.value == value:
                return True 
            
        return False 
    
    def get_child(self, value):
        
        for node in self.children:
            if node.value == value:
                return node 
            
        return None 
    
    def add_child(self, value): 
        
        child = FPNode(value, 1, self)
        self.children.append(child)
        
        return child
    
class FPTree(object):
    
    """
    A Frequent Pattern Tree
    """
    
    def __init__(self, transactions, threshold, root_value, root_count):
        
        self.frequent = self.find_frequence_items(transactions, threshold)
        self.headers = self.build_header_table(self.frequent)
        self.root = self.build_fptree(transactions, root_value, root_count,
                                      self.frequent, self.headers)
    
    @staticmethod
    def find_frequence_items(transactions, threshold):
        """
        Create a dictionary of item with occurances above the threshold

        Parameters
        ----------
        transations : TYPE
            DESCRIPTION.
        threshold : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        items = {}
        
        for transaction in transactions:
            for item in transaction:
                if item in items:
                    items[item] += 1
                else:
                    items[item] = 1
                    
        for key in list(items.keys()):
            if items[key] < threshold: 
                del items[key]
                
        return items
    
    @staticmethod
    def build_header_table(frequent):
        
        headers = {}
        for key in frequent.keys():
            headers[key] = None
        
        return headers
    
    def build_fptree(self, transactions, root_value, root_count, 
                     frequent, headers):
        
        root = FPNode(root_value, root_count, None)
        
        for transaction in transactions:
           sorted_item = [x for x in transaction if x in frequent]
           sorted_item.sort(key = lambda x: frequent[x], reverse = True)
           if len(sorted_item) > 0:
               self.insert_tree(sorted_item, root, headers)
               
        return root 
    
    def insert_tree(self, items, node, headers):
        
        first = items[0]
        child = node.get_child(first)
        if child is not None:
            child.count += 1
        else:
            child = node.add_child(first)
            
            if headers[first] is None:
                headers[first] = child
            else:
                current = headers[first]
                while current.link is not None:
                    current = current.link
                current.link = child
        
        remaining_items = items[1:]
        if len(remaining_items) > 0:
            self.insert_tree(remaining_items, child, headers)
            
    def tree_has_single_path(self, node):
        
        num_children = len(node.children)
        if num_children > 1:
            return False
        elif num_children == 0:
            return True 
        else:
            return True and self.tree_has_single_path(node.children[0])
        
    def mine_patterns(self, threshold):
        
        if self.tree_has_single_path(self.root):
            return self.generate_pattern_list()
        else:
            return self.zip_pattern(self.mine_sub_trees(threshold))
    
    def zip_pattern(self, patterns):
        
        suffix = self.root.value 
        
        if suffix is not None:
            new_patterns = {}
            for key in patterns.keys(): 
                new_patterns[tuple(sorted(list(key) + [suffix]))] = patterns[key]
            
            return new_patterns
        
        return patterns
                
    def generate_pattern_list(self):
        
        patterns = {}
        items = self.frequent.keys()
        
        if self.root.value is None:
            suffix_value = []
        else:
            suffix_value = [self.root.value]
            patterns[tuple(suffix_value)] = self.root.count 
        
        for i in range(1, len(items) + 1):
            for subset in itertools.combinations(items, i):
                pattern = tuple(sorted(list(subset) + suffix_value))
                pattern[pattern] = min([self.frequent[x] for x in subset])
                
        return patterns
    
    def mine_sub_trees(self, threshold):
        
        patterns = {}
        mining_order = sorted(self.frequent.keys(), 
                              key = lambda x: self.frequent[x])
        
        for item in mining_order:
            suffixes = []
            conditional_tree_input = []
            node = self.headers[item]
            
            # Follow node links to get a list of all occurances of a certain item
            while node is not None:
                suffixes.append(node)
                node = node.link
                
            # For each occurance of the item, trace the path back to root node
            for suffix in suffixes:
                
                frequency = suffix.count 
                path = []
                parent = suffix.parent 
                
                while parent.parent is not None:
                    path.append(parent.value)
                    parent = parent.parent
                    
                for i in range(frequency):
                    conditional_tree_input.append(path)
                    
            # Construct subtree and grad the pattern
                    
            subtree = FPTree(conditional_tree_input, threshold, 
                             item, self.frequent[item])
            subtree_patterns = subtree.mine_patterns(threshold)
            
            for pattern in subtree_patterns.keys():
                if pattern in patterns:
                    patterns[pattern] += subtree_patterns[pattern]
                    
                else:
                    patterns[pattern] = subtree_patterns[pattern]
                    
        return patterns
        
    
def find_frequence_patterns(transactions, support_threshold):
    
    tree = FPTree(transactions, support_threshold, None, None)
    return tree.mine_patterns(support_threshold)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    