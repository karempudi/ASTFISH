## Train script for channel segmentation network
from narsil2.segmentation.model_dev_old import trainNet
from torchvision import transforms
from narsil2.segmentation.transformations import changedtoPIL, randomCrop, randomRotation, randomAffine, normalize, toTensor, addnoise, padTo16
from narsil2.segmentation.transformations import randomVerticalFlip, shrinkSize
from datetime import datetime

rootDir = '../../../'

dataDir = rootDir + 'data/channelsData/'

species = ['channels']

date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

modelSavePath = rootDir + 'saved_models/channels_' + date + '.pth'


trainTransforms = transforms.Compose([changedtoPIL(), randomCrop(320), 
                                    #randomRotation([-20, 20]),
                                randomAffine((0.625, 1.5), None),
                                randomVerticalFlip(),
                                toTensor(),
                                normalize(),
                                addnoise(std=0.15)])
validationTransforms = transforms.Compose([changedtoPIL(), padTo16(),
                             toTensor(), normalize()])

modelParameters = {
    'netType': 'small',
    'transposeConv': True,
    'device': "cuda:1",
    'includeWeights': False
}

optimizationParameters = {
    'learningRate': 1e-3,
    'nEpochs': 10,
    'batchSize': 64,
    'cores': 6,
    'schedulerStep': 5,
    'schedulerGamma': 0.5
}

net = trainNet(dataDir, species, trainTransforms, modelParameters, optimizationParameters, validation=True, validationTransforms=validationTransforms)

net.train()

net.save(modelSavePath)
net.plotLosses(ylim=[0.0, 1.5])