# Computer Vision Project

This repository contains code and milestone materials for a computer vision project on few-shot generalized referring expression comprehension using gRefCOCO.

## Project Topic

The project studies whether frozen vision-language and region-aware representations can support generalized grounding over no-target, single-target, and multi-target referring expressions.

## Dataset

The main dataset is gRefCOCO, used together with MS COCO 2014 train images. The dataset files and images are not included in this repository due to size.

Expected local data structure:

```text
data/
├── grefcoco/
│   └── annotations/
│       ├── grefs(unc).json
│       └── instances.json
└── coco/
    └── train2014/

Repository Structure
src/        Data inspection, preprocessing, visualization, and DataLoader scripts
splits/     Few-shot split files
outputs/    Dataset statistics and generated figures

Milestone 1

Milestone 1 focuses on dataset inspection, preliminary preprocessing, few-shot subset construction, and DataLoader validation.
