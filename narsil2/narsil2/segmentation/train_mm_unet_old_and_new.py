
from narsil2.segmentation.model_dev_old import trainNet
from torchvision import transforms
from narsil2.segmentation.transformations import changedtoPIL, randomCrop, randomRotation, randomAffine, normalize, toTensor, addnoise, padTo16
from narsil2.segmentation.transformations import randomVerticalFlip, shrinkSize

from datetime import datetime

rootDir = '../../../'

dataDir = rootDir + 'data/unet_data/'

species = ['ecoli', 'mixed']

date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


#modelSavePath = rootDir + 'ASTPaper/segModels/test_pseudo.pth'
pretrain_model_path = rootDir + 'saved_models/unet_pretrain_only_ecoli.pth'
modelSavePath = rootDir + 'saved_models/unet_using_pretrain_mixed_old_new_' + date + '.pth'



trainTransforms = transforms.Compose([changedtoPIL(), randomCrop(320), 
                                    randomRotation([-20, 20]),
                                    #randomContrast(0.2),
                                randomAffine((0.625, 1.5), [-20, 20, -20, 20]),
                                randomVerticalFlip(),
                                toTensor(),
                                normalize(),
                                addnoise(std=0.20)])
validationTransforms = transforms.Compose([changedtoPIL(), shrinkSize(), padTo16(),
                             toTensor(), normalize()])


modelParameters = {
    'netType': 'big',
    'transposeConv': True,
    'device': "cuda:1",
    'includeWeights': True
}

optimizationParameters = {
    'learningRate': 1e-5,
    'nEpochs': 10,
    'batchSize': 8,
    'cores': 4,
    'schedulerStep': 2,
    'schedulerGamma': 0.5
}

net = trainNet(dataDir, species, trainTransforms, modelParameters, optimizationParameters, validation=True, validationTransforms=validationTransforms)

net.train(pretrain_model_path=pretrain_model_path)

net.save(modelSavePath)
net.plotLosses()