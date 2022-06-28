import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image

import models.crnn as crnn


model_path = './data/crnn.pth'
import sys
try:
    a=sys.argv[1]
    img_path=a
except:
    img_path = './data/demo.png'
# img_path = './data/demo.png'
alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'

model = crnn.CRNN(32, 1, 37, 256)
if torch.cuda.is_available():
    model = model.cuda()
print(('loading pretrained model from %s' % model_path))
model.load_state_dict(torch.load(model_path))

converter = utils.strLabelConverter(alphabet)

transformer = dataset.resizeNormalize((100, 32))
image = Image.open(img_path).convert('L')
image = transformer(image)
if torch.cuda.is_available():
    image = image.cuda()
image = image.view(1, *image.size())
image = Variable(image)
print('---- ', image.shape)

model.eval()
preds = model(image)
print('--------- ', preds.shape)
_, preds = preds.max(2)
print('------------ ', preds.shape)
preds = preds.transpose(1, 0).contiguous().view(-1)
print('--------------- ', preds.shape)

preds_size = Variable(torch.IntTensor([preds.size(0)]))
print('-------------------- ', preds_size.shape)
print('----------------------------- ', preds.data)
raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
print(('%-20s => %-20s' % (raw_pred, sim_pred)))
