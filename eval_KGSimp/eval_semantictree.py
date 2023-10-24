#!/usr/bin/python
#-*-coding:utf-8 -*-
#Author   : Zodiac
#Version  : 1.0
#Filename : parse_ctree.py
from __future__ import print_function

import sys

import stanza
import nltk
from nltk import Tree
from collections import defaultdict
from tqdm import tqdm


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Height:
    def __init(self):
        self.h = 0
 
# Optimised recursive function to find diameter
# of binary tree
def diameterOpt(root, height):
 
    # to store height of left and right subtree
    lh = Height()
    rh = Height()
 
    # base condition- when binary tree is empty
    if root is None:
        height.h = 0
        return 0
 
     
    # ldiameter --> diameter of left subtree
    # rdiameter  --> diameter of right subtree
     
    # height of left subtree and right subtree is obtained from lh and rh
    # and returned value of function is stored in ldiameter and rdiameter
     
    ldiameter = diameterOpt(root.left, lh)
    rdiameter = diameterOpt(root.right, rh)
 
    # height of tree will be max of left subtree
    # height and right subtree height plus1
 
    height.h = max(lh.h, rh.h) + 1
 
    # return maximum of the following
    # 1)left diameter
    # 2)right diameter
    # 3)left height + right height + 1
    return max(lh.h + rh.h + 1, max(ldiameter, rdiameter))
 

# function to calculate diameter of binary tree
def diameter(root):
    height = Height()
    d = diameterOpt(root, height)

    return height.h, d


def convertnary2bin(root):

    rootNode = TreeNode(root.label())

    queue = [(rootNode, root)]

    while queue:
        parent, cur = queue.pop(0)

        prevBNode = None
        headBNode = None

        if type(cur) == nltk.tree.Tree:
            for subtree in cur:
                newNode = TreeNode(subtree.label() if type(subtree) == nltk.tree.Tree else subtree)
                if prevBNode:
                    prevBNode.right = newNode
                else:
                    headBNode = newNode
                prevBNode = newNode
                queue.append((newNode, subtree))

        parent.left = headBNode

    return rootNode


def traverse_narytree(root):

    queue = [root]
    results = []

    while queue:
        node = queue.pop(0)

        results.append(node.label() if type(node) == nltk.tree.Tree else node)

        if type(node) == nltk.tree.Tree:
            for subtree in node:
                queue.append(subtree)

    print(sorted(results))


def traverse_btree(root):

    queue = [root]
    results = []

    while queue:
        node = queue.pop(0)
        results.append(node.val)

        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)

    print(sorted(results))


def main(file_name):
    nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')

    lines = []
    with open(file_name) as reader:
        for line in reader:
            lines.append(line.strip())
    for i, line in enumerate(tqdm(lines)):
        # if i > 3: break
        try:
            doc = nlp(line)
            children = []
            for sent in doc.sentences:
                node = sent.constituency
                tree = Tree.fromstring(str(node))
                children.append(tree)

            if len(children) == 1:
                root = children[0]
            else:
                root = Tree("ROOT", [child[0] for child in children])

            broot = convertnary2bin(root)
            height, diam = diameter(broot)
            # print(height, diam)
            print(i, height, diam)
        except:
            pass


if __name__ == "__main__":
    main(sys.argv[1])
