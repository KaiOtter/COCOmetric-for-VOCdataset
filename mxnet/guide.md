## Init metric class:
<hr>
** from metric import Custom_COCODetectionMetric**
val_metric = Custom_COCODetectionMetric(dataset, save_prefix, out_gt_json, resume=True,
                                            data_shape=(args.data_shape, args.data_shape))
----
dataset : instance of gluoncv dataset.
        The validation dataset.
save_prefix : str
        Prefix for the saved JSON results. (should be a complete path or only without extension)
gt_file : str
        Where to save the ground truth JSON (which is converted from VOC)
resume : bool
        whether re_eval on saved JSON results, if True, skip inference val_dataset (default False)
        
