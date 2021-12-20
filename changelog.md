# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
****

## [Unreleased]

## [0.0.1] 2021-12-16

- preprocess.make_list
  - 划分 train 和 val 数据集
- preprocess.apollo_to_coco_with_mask
  - 初步完成mask的提取，以及车辆遮挡区域的`直左`，`直右`等困难样本的合并
  - parallel processing
- preprocess.check_data
  - 按照类别输出箭头数据集，在原图的基础上绘制所在区域，并放大显示


