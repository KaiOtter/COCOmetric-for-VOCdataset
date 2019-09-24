# Using a customized Gluoncv VOCDetection as the dataset class

## Initialize metric class:

**from metric import Custom_COCODetectionMetric**

#### val_metric = Custom_COCODetectionMetric(dataset, save_prefix, out_gt_json, resume=True, data_shape=(256, 256))

*dataset : instance of gluoncv dataset.*

&ensp;&ensp;&ensp;&ensp;*The validation dataset.*

*save_prefix : str*

&ensp;&ensp;&ensp;&ensp;*Prefix for the saved JSON results. (should be a complete path or only without extension)*

*gt_file : str*

&ensp;&ensp;&ensp;&ensp;*Where to save the ground truth JSON (which is converted from VOC)*

*resume : bool*

&ensp;&ensp;&ensp;&ensp;*whether re_eval on saved JSON results, if True, skip inference val_dataset (default False)*
        
