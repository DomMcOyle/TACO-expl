"""
This file contains Functions and classes for TACO dataset loading in COCO 
format for the DETR based models.

Authors:
- Dell'Olio Domenico
- Delvecchio Giovanni Pio
- Disabato Raffaele
"""


import torch
import numpy as np
from .torchvision_datasets.coco import CocoDetection as TvCocoDetection
import HDDETR.datasets.transforms as T
from pycocotools import mask as coco_mask
import HDDETR.util.misc as mutils
from pathlib import Path

class TACODataset(TvCocoDetection):
    """
    Class modeling the TACO dataset using the COCO Format
    """
    def __init__(
        self,
        img_folder,
        ann_file,
        transforms,
        cache_mode=False,
        local_rank=0,
        local_size=1,
        use_crowd=False,
    ):
        """
        :param img_folder: path of the folder containing the images
        :param ann_file: path of the annotation file
        :param transforms: Functions to be applied for the dataset augmentation
        :param cache_mode: see docs for TVCocoDetection
        :param local_rank: see docs for TVCocoDetection
        :param local_size: see docs for TVCocoDetection
        :param use_crowd: boolean indicating whether to use crowd annotation or not
        """
        super(TACODataset, self).__init__(
            img_folder,
            ann_file,
            cache_mode=cache_mode,
            local_rank=local_rank,
            local_size=local_size,
        )
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(use_crowd)

    def __getitem__(self, idx):
        """
        function override of __getitem__
        :param idx: integer indicating the image to load
        :return : the pre processed image and the target information
        """
        img, target = super(TACODataset, self).__getitem__(idx)
        img = self.check_rotation_and_alpha(img)
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

    def check_rotation_and_alpha(self, image):
        """
        function checking whether the image must be rotated or its alpha channel must be removed
        :param image: PIL image to check
        :return : the rotated image
        """
        img_shape = np.shape(image)

        # load metadata
        exif = image.getexif()
        if exif:
            exif = dict(exif.items())
            # Rotate portrait images if necessary (274 is the orientation tag code)
            if 274 in exif:
                if exif[274] == 3:
                    image = image.rotate(180, expand=True)
                if exif[274] == 6:
                    image = image.rotate(270, expand=True)
                if exif[274] == 8:
                    image = image.rotate(90, expand=True)

        # If has an alpha channel, remove it for consistency
        if img_shape[-1] == 4:
            image = image[..., :3]
        return image


class ConvertCocoPolysToMask(object):
    """
    class modeling image pre-processing
    """
    def __init__(self, use_crowd):
       """
       :param use_crowd: boolean indicating whether to use crowd annotation or not
       """
       self.use_crowd = use_crowd

    def __call__(self, image, target):
        """
        :params image: image to be processed
        :params target: dictionart containing ground truth information
        """
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        if not self.use_crowd:
          anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        # condition as sanity check for empty segmentations
        boxes = [obj["bbox"] for obj in anno if obj["segmentation"]]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno if obj["segmentation"]]
        classes = torch.tensor(classes, dtype=torch.int64) - 1 # for labels between 0 and 9

        # to bitmap conversion
        segmentations = [obj["segmentation"] for obj in anno if obj["segmentation"]]
        masks = self.convert_coco_poly_to_mask(segmentations=segmentations, height=h, width=w)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        masks = masks[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["masks"] = masks
        target["image_id"] = image_id

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno if obj["segmentation"]])
        iscrowd = torch.tensor(
            [obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno if obj["segmentation"]]
        )
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)]) # original size...
        target["size"] = torch.as_tensor([int(h), int(w)]) # ...and placeholder for size after augmentation

        return image, target

    def convert_coco_poly_to_mask(self, segmentations, height, width):
      """
      Function converting the polygons obtained from the annotation file to bitmap
      :param segmentations: list of polygon representation for the segments
      :param height: image height
      :param width: image width
      """
      masks = []
      for polygons in segmentations:
          # converts polygon to compressed rle
          rles = coco_mask.frPyObjects(polygons, height, width)
          # decodes rle
          mask = coco_mask.decode(rles)
          if len(mask.shape) < 3:
            mask = mask[..., None]
          mask = torch.as_tensor(mask, dtype=torch.uint8)
          mask = mask.any(dim=2)
          masks.append(mask)
      if masks:
          masks = torch.stack(masks, dim=0)
      else:
          # if there are no segmentations a 0 dimensional mask is returned
          masks = torch.zeros((0, height, width), dtype=torch.uint8)
      return masks


def make_coco_transforms(image_set):
    """
    function creating transform objects depending on the splt.
    :param image_set: string indicating the split to generate ("train", "val", "test") or if only normalization must be added ("NORMALIZE_ONLY")
    """
    normalize = T.Compose(
        [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
    max_size = 1333
    #max_size_test = 2500
    #max_size = 1024
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
              # 832, 864, 896, 928, 960]

    if image_set == "train":
        return T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=max_size),
                    None,
                    #T.ZoomIn(max_size),
                    T.Compose(
                        [
                            # T.RandomResize([400, 500, 600])
                            T.RandomResize([2000, 2500, 3000]),
                            #T.RandomSizeCrop(384, 600),
                            T.RandomSizeCrop(1920, 3000),
                            T.RandomResize(scales, max_size=max_size),
                        ]
                    ),
                ),
                normalize,
            ]
        )
    if image_set == "val":
        return T.Compose([T.RandomResize([800], max_size=max_size), normalize,])
    if image_set == "test":
        return T.Compose([T.RandomResize([800], max_size=max_size), normalize,])
    if image_set == "NORMALIZE_ONLY":
      return normalize

    raise ValueError(f"unknown {image_set}")

def create_dataset(split, add_transform=True):
  """
  function creating the dataset object
  :param split: string indicating the split to be considered ("train", "val", "test")
  :param add_transform: boolean indicating whether to add augmentation or just normalization
  :return : an appropriate TACODataset object
  """
  ann_file = Path("/content/TACO-expl/data/annotations_off_0_" + split +".json")
  img_folder = Path("/content/MyDrive/MyDrive/official/")

  if not add_transform:
    trans = make_coco_transforms("NORMALIZE_ONLY")
  else:
    trans = make_coco_transforms(split)
  dataset = TACODataset(
      img_folder,
      ann_file,
      transforms=trans,
      local_rank=mutils.get_local_rank(),
      local_size=mutils.get_local_size(),
  )
  return dataset