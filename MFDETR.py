"""
This file contains the model declaration and the helper function to load Mask-Frozen DETR. The code is taken from MFDETR Training Notebook.ipynb.

Authors:
- Dell'Olio Domenico
- Delvecchio Giovanni Pio
- Disabato Raffaele
"""

from HDDETR.models.deformable_transformer import DeformableTransformerEncoderLayer, DeformableTransformerEncoder
import HDDETR.models
from HDDETR.models.deformable_detr import MLP
from HDDETR.models.position_encoding import PositionEmbeddingSine
from torchvision.ops import RoIAlign
import HDDETR.util.misc as mutils
from HDDETR.util import box_ops
from HDDETR.datasets.taco_dataset import create_dataset
from HDDETR.configs.attr_dict import get_DETRargs
import torch.nn as nn
import torch

def load_detr(args, num_classes, finetuning=False, load_dict="r50.pth"):
  """
  Function loading the DETR model.
  :params args: AttrDict object specifying the model hyperparameters
  :params num_classes: number of classes
  :params finetuning: boolean indicating whether the model is loaded for finetuning or not
  :load_dict: path of the detr checkpoint to load
  :return : the model object, the model training criterion (loss computing object) and the image postprocessors
  """
  model,crit, postproc = HDDETR.models.build(args)
  crit.num_classes = num_classes

  if finetuning:
      # if the model has to be finetuned, then the class embedding layers weights must be deleted
      state_dict = torch.load(load_dict)["model"]
      to_del = []
      for k in state_dict.keys():
        if "class_embed" in k:
          to_del.append(k)
      for k in to_del:
        del state_dict[k]
  else:
      # if the model has already been finetuned, then it must be prepared for
      # the segmentation network training
      if load_dict is not None:
        state_dict = torch.load(load_dict)["model_state_dict"]
      model.num_queries = 300
      model.transformer.two_stage_num_proposals = 300
      postproc.update({"segm": HDDETR.models.segmentation.PostProcessSegmMFD()})
      postproc.update({"bbox": HDDETR.models.deformable_detr.PostProcessMFD()})
      crit.losses.append("masksMFD")
      crit.losses.remove("boxes")
      crit.losses.remove("labels")
      crit.weight_dict["loss_mask"] = 1
      crit.weight_dict["loss_dice"] = 1
      crit.weight_dict["loss_mask_score"] = 1
  if load_dict is not None:
    print("***LOADING DICT***")
    model.load_state_dict(state_dict, strict=False)
  return model, crit, postproc

class MaskFrozenDETR(nn.Module):
  """
  Mask-Frozen DETR class
  """
  def __init__(self, detr, device, num_classes, box_channels=128):
    """
    :param detr: detr instance to use as frozen feature and proposal extractor
    :param device: torch device on which the model must operate
    :param num_classes: number of classes to predict
    :param box_channels: channels to use for the extracted box activation maps
    """
    super().__init__()
    self.device = device
    self.detr = detr
    self.detr.num_queries = self.detr.num_queries_one2one
    self.detr.transformer.two_stage_num_proposals = self.detr.num_queries_one2one
    self.num_classes = num_classes
    for param in detr.parameters():
      param.requires_grad = False
    # see deformable encoder block
    pre_compression_channels = 256
    # layer for feature encoding
    feature_enc_layer = DeformableTransformerEncoderLayer(d_model=pre_compression_channels,dropout=0, activation='gelu')
    # layer for boxed feature ancoding
    box_enc_layer = DeformableTransformerEncoderLayer(d_model=box_channels,dropout=0, activation='gelu')
    self.feature_enc = DeformableTransformerEncoder(feature_enc_layer, 2)
    self.box_enc = DeformableTransformerEncoder(box_enc_layer, 2)
    self.box_pos_embedding = PositionEmbeddingSine(box_channels//2, normalize=True)
    # resizes  feature maps before further processing
    self.channel_mapper = nn.Conv2d(pre_compression_channels, box_channels, kernel_size=1)
    # maps the features back to the original number of channels
    self.query_channel_mapper = nn.Linear(pre_compression_channels, box_channels)
    self.roialign = RoIAlign(output_size=(32,32),spatial_scale=0.25, sampling_ratio=-1)

    self.neck = nn.Conv2d(96, pre_compression_channels, kernel_size=1)
    self.neck_gn = nn.GroupNorm(pre_compression_channels//32,pre_compression_channels)

    self.class_adapter = nn.Linear(pre_compression_channels, num_classes)
    self.topk = 128 # number of boxes to keep

    self.maskiouhead = MaskIOUHead(box_channels + 1)



  def forward(self, input):
    bs, _, h, w = input.tensors.shape
    # get output from the H-DETR
    detr_out = self.detr(input)

    # pass the multi-scale encoder maps to the two deformable layers
    enc_maps, _ = self.feature_enc(**detr_out["intermediate_enc_out"])

    # interpolate the maps to the backbone dimension
    backbone_h, backbone_w = detr_out["backbone_out"].tensors.shape[-2], detr_out["backbone_out"].tensors.shape[-1]
    map_limit = detr_out["intermediate_enc_out"]["level_start_index"][1].item()
    enc_map_h = detr_out["intermediate_enc_out"]["spatial_shapes"][0,0].item()
    enc_map_w = detr_out["intermediate_enc_out"]["spatial_shapes"][0,1].item()
    enc_maps = enc_maps[:,:map_limit,:].reshape(bs, enc_map_h, enc_map_w, 256).permute(0,3,1,2)
    fe = mutils.interpolate(enc_maps, (backbone_h,backbone_w), mode="bilinear") # shape: (bs, ch, h, w)

    # sum the maps and reduce dimensionality
    c1 = self.neck_gn(self.neck(detr_out["backbone_out"].tensors)) # shape (bs,ch, h, w)
    f = c1 + fe

    # map channel reduction
    mapped_f = self.channel_mapper(f)

    # computing class for each proposal
    logits = detr_out['pred_logits']

    # computing top ~100 proposals, boxes and queries
    ci = logits.sigmoid()
    topk_values, topk_indexes = torch.topk(
            ci.view(logits.shape[0], -1), self.topk, dim=1
    ) # takes top logits. may take a box more than once
    nboxes = detr_out["pred_boxes"]
    topk_boxes = topk_indexes // logits.shape[2]
    nboxes = torch.gather(nboxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
    nboxes = nboxes[:,:self.topk, :]
    boxes = box_ops.box_cxcywh_to_xyxy(nboxes)
    img_h, img_w = input.sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    boxes = boxes * scale_fct[:, None, :]
    boxes = torch.clamp(boxes, min=0)
    object_queries = detr_out["decoder_out"][-1, :, :self.detr.num_queries_one2one, :]
    object_queries = torch.gather(object_queries, 1, topk_boxes.unsqueeze(-1).repeat(1,1,object_queries.shape[-1]))
    logits = torch.gather(logits, 1, topk_boxes.unsqueeze(-1).repeat(1,1, self.num_classes))

    # cut boxes
    batchindexes = torch.arange(bs).reshape(bs,1,1).repeat(1,self.topk,1).to(self.device)
    roiboxes = torch.cat([batchindexes, boxes], -1).reshape(bs*self.topk, 5)

    Ri = self.roialign(mapped_f, roiboxes).permute(0, 2, 3, 1).reshape(bs*self.topk, 32, 32, 128)
    maskRi = self.roialign(detr_out["backbone_out"].mask.to(torch.float32).unsqueeze(1), roiboxes)\
                 .to(torch.bool)\
                 .permute(0,2,3,1).squeeze().reshape(bs*self.topk, 32,32)
    # provess boxes
    box_pos = self.box_pos_embedding(mutils.NestedTensor(Ri, maskRi)).reshape(self.topk*bs, 32*32, 128)


    valid_ratios_box = torch.stack([self.detr.transformer.get_valid_ratio(m.unsqueeze(0)) for m in maskRi], 0)

    sp_shapes = torch.tensor([[32,32]]).to(self.device)
    level_start_index = torch.tensor([0]).to(self.device)
    Ri = Ri.reshape(bs*self.topk,32*32, 128)
    maskRi = maskRi.reshape(bs*self.topk,32*32)

    Ri, _ = self.box_enc(Ri,
                      padding_mask=maskRi,
                      spatial_shapes=sp_shapes,
                      pos=box_pos,
                      level_start_index=level_start_index,
                      valid_ratios=valid_ratios_box)


    object_queries = self.query_channel_mapper(object_queries).reshape(bs*self.topk, 128, 1)


    segmasks = torch.bmm(Ri, object_queries).reshape(bs*self.topk, 1, 32, 32)
    # evaluates the scores of the masks
    mask_scores = self.maskiouhead(torch.cat([segmasks.sigmoid(),
                                          Ri.reshape(bs*self.topk,32,32, 128).permute(0,3,1,2)], 1)).reshape(bs, self.topk)
    segmasks = segmasks.reshape(bs, self.topk, 32, 32)
    return {"pred_masks": segmasks,
            "pred_logits": logits,
            "pred_boxes": nboxes,
            "unnormal_boxes": boxes,
            "top_scores": topk_values,
            "mask_scores": mask_scores,
            "top_indexes": topk_indexes}

class MaskIOUHead(nn.Module):
  """
  MaskIoU prediction head
  """
  def __init__(self, input_ch=129):
    """
    :param input_ch: input channels. must be set to the number of channels of
                     the boxed feature map after the encoding layer + 1
    """
    super().__init__()
    self.conv1 = nn.Conv2d(input_ch, 128, 3, 1, 1)
    self.conv2 = nn.Conv2d(128, 128, 3, 1, 1)
    self.conv3 = nn.Conv2d(128, 128, 3, 2, 1)
    self.conv4 = nn.Conv2d(128, 128, 3, 2, 1)
    self.mlp = MLP(128*8*8, 1024, 1, 3)
    self.relu = nn.ReLU()

  def forward(self, x):
    h = self.relu(self.conv1(x))
    h = self.relu(self.conv2(h))
    h = self.relu(self.conv3(h))
    h = self.relu(self.conv4(h))
    return self.mlp(h.flatten(1))