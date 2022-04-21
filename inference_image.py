import segmentation_models_pytorch as smp
import torch
from torchvision import transforms as TF
import numpy as np
from PIL import Image
from collections import OrderedDict

ENCODER = 'resnet18'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = 1
ATTENTION = None
ACTIVATION = None
DEVICE = 'cuda:1'
to_tensor = TF.ToTensor()
model = smp.Unet(encoder_name=ENCODER,
                 encoder_weights=ENCODER_WEIGHTS,
                 classes=CLASSES,
                 activation=ACTIVATION)

weights = torch.load('./epoch_16_best.ckpt')
new_weights = OrderedDict()
for key in weights.keys():
    new_key = '.'.join(key.split('.')[1:])
    new_weights[new_key] = weights[key]

model.load_state_dict(new_weights)
model.to(DEVICE)
model.eval()

#inference image
img_pth = './Dataset/teresa_02_00031_0.jpg'
res_pth = './Dataset/teresa_02_00031_0_mask.jpg'
I = Image.open(img_pth)
input_image = to_tensor(I).unsqueeze(0)
input_image = input_image.to(DEVICE)
print(f"input_image {input_image.shape} {type(input_image)}")
with torch.no_grad():
    pred = model(input_image)
    torch.cuda.synchronize()
    pred_mask = (pred > 0).type(torch.int8)
    pred_mask = pred_mask.squeeze().cpu().numpy()
    print(f"pred_mask {pred_mask.shape} {type(pred_mask)}")
    result = (pred_mask * 255.0).astype(np.uint8)
    Image.fromarray(result).save(res_pth)