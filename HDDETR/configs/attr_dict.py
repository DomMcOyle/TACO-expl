import numpy as np
class AttrDict(dict):
    """
    class modelling the command line arguments
    """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        
def get_DETRargs():
    """
    Function producing an attribute dictionary containing the 
    command line arguments to initialize a DETR instance
    """
    args = AttrDict()
    args.output_dir = './out/'
    args.with_box_refine = True # adding box refinement
    args.two_stage = True # considering two stage detectro
    args.dim_feedforward = 2048 # dimension of the feedforward network in the encoder/decoder layers
    args.num_queries_one2one = 300 # number of queries to consider for the one2one matching
    args.num_queries_one2many = 1500 # number of queries to consider for the one2many matching
    args.k_one2many = 6 # multiplicity of the one2many matching
    args.lambda_one2many = 1.0 # weight of the one2many matching loss
    args.mixed_selection = True # a trick for Deformable DETR two stage
    args.look_forward_twice = True # performs double look forward for reference points
    args.dataset_file = "coco"
    args.device = 'cuda'
    args.hidden_dim = 256 # MLP hidden dimension
    args.position_embedding = 'sine' # kind of position embedding
    args.position_embedding_scale = np.pi *2
    args.lr_backbone =1e-5
    args.backbone = "swin_tiny"
    args.pretrained_backbone_path = None
    args.masks = False # if masks must be produced
    args.num_feature_levels = 4 #levels for multi scale deformable attention
    args.dilation = False
    args.nheads = 8
    args.enc_layers = 6
    args.dec_layers = 6
    args.dim_feedforwards = 2048
    args.dropout = 0
    args.dec_n_points = 4 # number of points for deformable attention computation
    args.enc_n_points = 4
    args.use_checkpoint = True
    args.aux_loss = True # if loss must be computed also on intermediate layers
    args.cls_loss_coef=2 # classification scale loss coefficient
    args.giou_loss_coef=2 # giou loss scale coefficient
    args.maskiou_loss_coef=1 # maskiou loss scale coefficient
    args.focal_alpha=0.25
    args.topk=100 #topk for evaluation
    # coefficients for hungarian matcher
    args.bbox_loss_coef=5 #5
    args.set_cost_class=2
    args.set_cost_bbox=5
    args.set_cost_giou=2
    args.num_classes = 10
    args.drop_path_rate = 0.2
    return args