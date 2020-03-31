# !/usr/bin/env python
# -*- coding: utf-8 -*-

# This script is to modify the align_corners parameter of the ResizeBilinear operator,
# and split the avgpool operator into 2 cascadede avgpool operators.
# It was tested to work with tf 1.14 version on ubuntu system

import os
import numpy as np
import tensorflow as tf


originModelPath = './deeplab_v3_mobilenet_v2_quant.pb'
pbModelOutputNodeList = ['ResizeBilinear_3']

modifyModelPath = './deeplab_v3_mobilenet_v2_quant_modify.pb'
targetNodeName = 'AvgPool2D/AvgPool'

def loadGraph(frozen_graph_filepath):
    with tf.gfile.GFile(frozen_graph_filepath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
    return graph

def saveGraph(dstGraph, dstModelPath):
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(dstGraph, name='')
    with tf.Session(graph=graph) as sess:
        constGraph = tf.graph_util.convert_variables_to_constants(sess,
              sess.graph_def, pbModelOutputNodeList)
    with tf.gfile.FastGFile(dstModelPath, mode='wb') as f:
        f.write(constGraph.SerializeToString())
    pass

def modifyGraph(srcGraph):
    dstGraph = tf.GraphDef()
    with tf.Session(graph=srcGraph) as sess:
        for srcNodeItem in sess.graph_def.node:
            if srcNodeItem.op == 'ResizeBilinear':
                changeResizeBilinear(dstGraph, srcNodeItem)
                continue
            if srcNodeItem.name == targetNodeName:
                splitAvgPoolParam(dstGraph, srcNodeItem)
                continue
            dstNodeItem = dstGraph.node.add()
            dstNodeItem.CopyFrom(srcNodeItem)
        pass
    return dstGraph

def changeResizeBilinear(dstGraph, srcNode):
    dstNode = dstGraph.node.add()
    dstNode.op = srcNode.op
    dstNode.name = srcNode.name
    dstNode.attr['align_corners'].b = False
    if 'half_pixel_centers' in srcNode.attr:
        dstNode.attr['half_pixel_centers'].CopyFrom(tf.AttrValue(b=srcNode.attr['half_pixel_centers'].b))
    if 'T' in srcNode.attr:
        dstNode.attr['T'].CopyFrom(tf.AttrValue(type=srcNode.attr['T'].type))
    for inputItem in srcNode.input:
        dstNode.input.extend([inputItem])
    pass

def splitAvgPoolParam(dstGraph, srcNode):
    dstNodePart1 = dstGraph.node.add()
    newAddNodeName = '{:s}Part1'.format(srcNode.name)
    initAvgPoolNode(dstNode=dstNodePart1,
                    srcNode=srcNode,
                    newName=newAddNodeName,
                    newInput=srcNode.input,
                    newStrides=[1, 11, 11, 1],
                    newKsize=[1, 11, 11, 1])
    dstNodePart2 = dstGraph.node.add()
    initAvgPoolNode(dstNode=dstNodePart2,
                    srcNode=srcNode,
                    newName=srcNode.name,
                    newInput=[newAddNodeName],
                    newStrides=[1, 1, 1, 1],
                    newKsize=[1, 3, 3, 1])
    pass

def initAvgPoolNode(dstNode, srcNode, newName, newInput, newStrides, newKsize):
    dstNode.op = 'AvgPool'
    dstNode.name = newName
    for inputItem in newInput:
        dstNode.input.extend([inputItem])
    if 'T' in srcNode.attr:
        dstNode.attr['T'].CopyFrom(tf.AttrValue(type=srcNode.attr['T'].type))
    if 'data_format' in srcNode.attr:
        dstNode.attr['data_format'].CopyFrom(tf.AttrValue(s=srcNode.attr['data_format'].s))
    if 'padding' in srcNode.attr:
        dstNode.attr['padding'].CopyFrom(tf.AttrValue(s=srcNode.attr['padding'].s))
    dstNode.attr['strides'].CopyFrom(tf.AttrValue(list=tf.AttrValue.ListValue(i=newStrides)))
    dstNode.attr['ksize'].CopyFrom(tf.AttrValue(list=tf.AttrValue.ListValue(i=newKsize)))
    pass

if __name__ == '__main__':
    srcGraph = loadGraph(originModelPath)
    dstGraph = modifyGraph(srcGraph)
    saveGraph(dstGraph, modifyModelPath)