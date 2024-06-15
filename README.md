# REPOSITORY FOR THE ML4CV PROJECT @ UNIBO A.Y. 22/23
<img src="https://raw.githubusercontent.com/wiki/pedropro/TACO/images/logonav.png" width="25%"/>

Link for the dataset: https://drive.google.com/drive/folders/1UqMT7axH7pZy8eydiOT90Vi_6DaNK4i-?usp=sharing

Link for the resized dataset: https://drive.google.com/drive/folders/1okE6U76brqBu-skV11iAzLYgNCX9IC1s?usp=sharing

Link to MFDETR model weights: https://drive.google.com/drive/folders/12zwQl0ijFbmulIa_fW-9_Xmx4BuGGoCg?usp=drive_link

Link to MaskDINO model weights: https://drive.google.com/drive/folders/1_4FLKe_NkUncGklEuIjqRbWVoDeJcWDy?usp=drive_link


Repository structure:
Folders:
- AdelaiDet: contains libraries and code for SOLOv2
- data: contains the full annotation, the split and the resized ones for the TACO dataset
- detector: contains the code used by the TACO authors to benchmark a instance segmentation model (Mask R-CNN)
- HDDETR: contains libraries and code for Mask-Frozen DETR and DETR.
- MaskDINO: contains libraries and code for MaskDINO 
- maskdino_config: contains MaskDINO config files
- res: contains model diagrams, MaskDINO training stats and test set predictions and ground truth visualization
- solov2_config: contains SOLOv2 config files

Files:
- Dataset splitting ad label replacing.ipynb: contains the script to split the official TACO dataset and get the TACO-10 subset of labels
- download.py: contains the script for the dataset downloading
- Detection-based XAI.ipynb: contains the script for object detection based explainability methods
- MaskDINO Training Notebook.ipynb: contains the training script for the MaskDINO model training
- MFDETR Training Notebook.ipynb: contains the training script for the Mask-Frozen DETR model training
- MFDETR.py: contains the Mask-Frozen DETR model implementation and helper functions for loading
- README.md: this file
- requirements.txt: requirements file 
- SOLOv2 Training Notebook: contains the training script for the SOLOv2 model
- validation resizing.ipynb: contains functions for the dataset resizing (images and targets) as well as for image rotation
- visual_utils.py: contains functions for segmentation-based XAI


