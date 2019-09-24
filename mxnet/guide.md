# Using a customized Gluoncv VOCDetection as the dataset class

## Initialize metric class:

**from metric import Custom_COCODetectionMetric**

### val_metric = Custom_COCODetectionMetric(dataset,&ensp;save_prefix,&ensp;out_gt_json,&ensp;resume=False,&ensp;data_shape=(256, 256))

*[dataset]() : instance of gluoncv dataset.*

&ensp;&ensp;&ensp;&ensp;*The validation dataset.*

*[save_prefix]() : str*

&ensp;&ensp;&ensp;&ensp;*Prefix for the saved JSON results. (should be a complete path or only without extension)*

*[gt_file]() : str*

&ensp;&ensp;&ensp;&ensp;*Where to save the ground truth JSON (which is converted from VOC)*

*[resume]() : bool*

&ensp;&ensp;&ensp;&ensp;*whether re_eval on saved JSON results, if True, skip inference val_dataset (default False)*


## Feed prediction :
*[pred_bboxes]() : mxnet.NDArray or numpy.ndarray

&ensp;&ensp;&ensp;&ensp;*Prediction bounding boxes with shape `B, N, 4`.

&ensp;&ensp;&ensp;&ensp;*Where B is the size of mini-batch, N is the number of bboxes.

*[pred_labels]() : mxnet.NDArray or numpy.ndarray

&ensp;&ensp;&ensp;&ensp;*Prediction bounding boxes labels with shape `B, N`.

*[pred_scores]() : mxnet.NDArray or numpy.ndarray

&ensp;&ensp;&ensp;&ensp;*Prediction bounding boxes scores with shape `B, N`.

### eval_metric.update(pred_boxes,&ensp;pred_labels,&ensp;pred_scores)

## Get eval results and print:

### map_name, mean_ap = eval_metric.get()
### from metric import print_eval
### print_eval(map_name,&ensp;mean_ap)

# Modify on Dataset Class
#### add a function for get original image shape by [img_id]()
#### [img_id]() is a string of image name without extension
#### and [self._im_shapes]() is dict() which should be filled during loading labels. (VOC xml files)
```javascript
def get_ori_shape(self, img_id):
        width, height = self._im_shapes[img_id]
        return width, height
```

#### add a function for exporting VOC gt_anno into a JSON in coco style
#### this should be called after val_dataset init and before val_metric init
```javascript
def export_coco_json(self, json_file):
        # only support load gt_bbox info
        if len(self._items) == 0:
            raise RuntimeError('should load items before exporting')
        import json

        import datetime
        info = {
            "year": 2019,
            "version": '1.0',
            "description": 'Convert custom VOCformat anno to COCO Json',
            "contributor": 'keyboardman',
            "url": 'none',
            "date_created": str(datetime.datetime.now().replace(microsecond=0)),
        }

        json_dict = {"info": info,
                     "images": [],
                     "annotations": [],
                     "categories": []
                     }
        box_id = 0

        for i in range(len(self._items)):
            img_id = self._items[i][1]
            if not img_id.isdigit():
                raise ValueError("only support using a int number as file name")
            labels = self._load_label(i)
            width, height = self.get_ori_shape(img_id)
            image = {'id': int(img_id),
                     'width': width,
                     'height': height,
                     'file_name': "{}.jpg".format(img_id),
                     }

            json_dict['images'].append(image)

            for label in labels:
                # tmp = [xmin, ymin, xmax, ymax, cls_id]
                # self.index_map[cls_name]
                ann = {
                    "id": box_id,
                    "image_id": int(img_id),
                    "category_id": int(label[4]),
                    "segmentation": [],
                    "area": width * height,
                    "bbox": [label[0], label[1], label[2] - label[0], label[3] - label[1]],
                    "iscrowd": 0,
                }
                json_dict['annotations'].append(ann)
                box_id += 1

        for k in self.index_map.keys():
            cat = {'id': int(self.index_map.get(k)),
                   'name': k,
                   'supercategory': 'none',
                   }
            json_dict['categories'].append(cat)

        json_fp = open(json_file, 'w')
        json_str = json.dumps(json_dict)
        json_fp.write(json_str)
        json_fp.close()
```
