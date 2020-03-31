# !/usr/bin/env python
# -*- coding: ytf-8 -*-

# This script is to modify the strides parameters of the avgpool operator in the model.
# It was tested to work with tf 1.14 version on ubuntu system

import os
import numpy as np
import tensorflow as tf

originModelPath = './deeplab_v3_mobilenet_v2_fp32.pb'
pbModelOutputNodeList = ['ResizeBilinear_3']

modifyModelPath = './deeplab_v3_mobilenet_v2_fp32_modify.pb'
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
            if srcNodeItem.name == targetNodeName:
                changeAvgPoolParam(dstGraph, srcNodeItem)
                continue
            dstNodeItem = dstGraph.node.add()
            dstNodeItem.CopyFrom(srcNodeItem)
        pass
    return dstGraph

def changeAvgPoolParam(dstGraph, srcNode):
    dstNode = dstGraph.node.add()
    dstNode.op = srcNode.op
    dstNode.name = srcNode.name
    for inputItem in srcNode.input:
        dstNode.input.extend([inputItem])
    if 'T' in srcNode.attr:
        dstNode.attr['T'].CopyFrom(tf.AttrValue(type=srcNode.attr['T'].type))
    if 'data_fromat' in srcNode.attr:
        dstNode.attr['data_format'].CopyFrom(tf.AttrValue(s=srcNode.attr['data_format'].s))
    if 'ksize' in srcNode.attr:
        dstNode.attr['ksize'].CopyFrom(tf.AttrValue(list=srcNode.attr['ksize'].list))
    if 'padding' in srcNode.attr:
        dstNode.attr['padding'].CopyFrom(tf.AttrValue(s=srcNode.attr['padding'].s))
    if 'strides' in srcNode.attr:
        dstNode.attr['strides'].CopyFrom(tf.AttrValue(list=tf.AttrValue.ListValue(i=[1, 1, 1, 1])))
    pass

if __name__ == '__main__':
    srcGraph = loadGraph(originModelPath)
    dstGraph = modifyGraph(srcGraph)
    saveGraph(dstGraph, modifyModelPath)