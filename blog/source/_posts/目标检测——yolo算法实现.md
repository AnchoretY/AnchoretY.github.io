---
title: 目标检测——yolo算法实现
date: 2019-11-05 15:37:25
tags:
---



​	本文主要针对yolo模型的实现细节进行一些记录，前半部分不讨论yolo模型，直接将yolo模型当做黑盒子进行使用，讨论模型外的预处理以及输出表现等部分。



### Yolo外部关键点

#### 1.boxes阈值过滤

​	该部分主要用于将**对各个boxes进行打分，根据阈值对boxes进行过滤**。

~~~python
def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6):
    box_scores = box_confidence*box_class_probs     #boxes分数为置信度*类别概率
    box_classes = K.argmax(box_scores,-1)
    box_class_scores = K.max(box_scores,-1)
    filtering_mask = box_class_scores>threshold
    scores = tf.boolean_mask(box_class_scores,filtering_mask)
    boxes = tf.boolean_mask(boxes,filtering_mask)
    classes = tf.boolean_mask(box_classes,filtering_mask)
 

~~~



bbox信息（x,y,w,h）为**物体中心位置相对于格子位置**的偏移、高度和宽度，均被归一化

置信度反映了是够包含物体以及包含物体情况下的为位置准确性，定义为