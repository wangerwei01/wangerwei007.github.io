---
layout:     post
title:      xml文件的python解析
subtitle:   五 一节后的代码狂欢
date:       2019-05-06
author:     WEW
header-img: img/post-bg-art.jpg
catalog: true
tags:
    - Python
    - 工具
    - 语言
---
# xml介绍
它是一种简洁的数据存储格式，本文针对xml的解析进行了相关总结。python基于xml的解析主要有三种方法，包含SAX、DOM和ElementTree，本文就后两种方法进行讲解。
# xml解析
## ElementTree方法
以下列xml文件为例

    <annotation>
    <folder>Non_Motor_Data</folder>
    <filename>PD_traffic_xuhui-image_00002-436.jpg</filename>
    <size>
        <width>1920</width>
        <height>1080</height>
        <depth>3</depth>
    </size>
    <object>
        <name>person</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>499</xmin>
            <ymin>121</ymin>
            <xmax>582</xmax>
            <ymax>287</ymax>
        </bndbox>
    </object>
    <object>
        <name>person</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>1193</xmin>
            <ymin>276</ymin>
            <xmax>1290</xmax>
            <ymax>508</ymax>
        </bndbox>
    </object>
    <object>
        <name>person</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>59</xmin>
            <ymin>23</ymin>
            <xmax>108</xmax>
            <ymax>139</ymax>
        </bndbox>
    </object>
    </annotation>
    
首先需要引入包
    
    from xml.etree.ElementTree import ElementTree, Element
    
具体的我们可以通过该函数进行解析

    def read_xml(xml_path):
        tree=ElementTree()#建立一个有效树
        tree.parse(xml_path)#解析xml文件
        return tree

在获取xml解析树之后我们可以采用层级遍历该树形结构

    def read_tree(tree):
        root=tree.getroot()#获取tree的根节点
        for child in root：
            if(child.tag=='size'):
                for subchild in child:
                    if(subchild.tag=='width'):
                        width=subchild.text
                    if(subchild.tag=='height'):
                        height=subchild.text
                        
通过以上方法可以层级对xml文件进行遍历

## DOM方法解析

以下面xml文件为例

    <collection shelf="New Arrivals">
        <movie title="Enemy Behind">
           <type>War, Thriller</type>
           <format>DVD</format>
           <year>2003</year>
           <rating>PG</rating>
           <stars>10</stars>
           <description>Talk about a US-Japan war</description>
        </movie>
        <movie title="Transformers">
           <type>Anime, Science Fiction</type>
           <format>DVD</format>
           <year>1989</year>
           <rating>R</rating>
           <stars>8</stars>
           <description>A schientific fiction</description>
        </movie>
        <movie title="Trigun">
           <type>Anime, Action</type>
           <format>DVD</format>
           <episodes>4</episodes>
           <rating>PG</rating>
           <stars>10</stars>
           <description>Vash the Stampede!</description>
        </movie>
      </collection>
      
 首先也需要引入包
 
     from xml.dom.minidom import parse
     import xml.dom.minidom
     
 进行xml解析
 
     DOMTree=xml.dom.minidom.parse(xml_path)
     collection=DOMTree.documentElement#获取文档集合
     if collection.hasAttribute("shelf"):
           print ("Root element : %s" % collection.getAttribute("shelf"))
      movies = collection.getElementsByTagName("movie")
      for movie in movies:
          print ("*****Movie*****")
          if movie.hasAttribute("title"):
              print ("Title: %s" % movie.getAttribute("title"))
          type = movie.getElementsByTagName('type')[0]
          print ("Type: %s" % type.childNodes[0].data)
          format = movie.getElementsByTagName('format')[0]
          print ("Format: %s" % format.childNodes[0].data)
          rating = movie.getElementsByTagName('rating')[0]
          print ("Rating: %s" % rating.childNodes[0].data)
          description = movie.getElementsByTagName('description')[0]
          print ("Description: %s" % description.childNodes[0].data)
          
 输出
     
    Root element : New Arrivals
    *****Movie*****
    Title: Enemy Behind
    Type: War, Thriller
    Format: DVD
    Rating: PG
    Description: Talk about a US-Japan war
    *****Movie*****
    Title: Transformers
    Type: Anime, Science Fiction
    Format: DVD
    Rating: R
    Description: A schientific fiction
    
# 生成xml文件

首先引入包
    
    from lxml.etree import Element, SubElement, tostring
    #from xml.dom.minidom import parseString 这个可有可无
    
建立xml其实也很简单，也就是层级进行构建，以下语句是建立xml文件的关键点

    node_root=Element('annotation')#建立根节点(建立一个节点)
    node_floder = ```SubElement(node_root,'floder')```#根节点下创立子节点floder
    node_floder.text = 'Non_Motor_Data'# 赋予子节点值为Non_Motor_Data
    

    
