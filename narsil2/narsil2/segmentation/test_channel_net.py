
from narsil2.segmentation.model_dev_old import testNet
from narsil2.segmentation.transformations import tensorizeOneImage, resizeOneImage, UnetTestTransformations
from torchvision import transforms

rootDir = '../../../../'


modelSavePath = '../../../saved_models/channels.pth'
phaseDir = rootDir + 'EXP-20-BV6129 AST FISH 201117/The run/Pos101/phaseFast/'
saveDir = None
fileformat = '*.tiff' 

#resizing = resizeOneImage((1040, 2048), (1040, 2048))
#resizing = resizeOneImage((2048, 4096), (1024, 2048))
#tensorizing = tensorizeOneImage(1)

imgTransforms = transforms.Compose([UnetTestTransformations(return_tensors=True)]) 

threshold = 0.9
testNet(modelSavePath, phaseDir, saveDir, imgTransforms, 0.8, fileformat=fileformat, 
		plotContours=True, contoursThreshold=0.85, removeSmallObjects=True)