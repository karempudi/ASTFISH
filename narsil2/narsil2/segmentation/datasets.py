
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pathlib
from pathlib import Path
from skimage import io
from skimage.transform import resize, rotate
from skimage.filters import gaussian
from narsil2.segmentation.transformations import OmniTransformations, padNumpyArray
from torchvision import transforms
from narsil2.segmentation.utils_omni import labels_to_flows_cpu_omni
import numpy as np
import torch
import edt
import glob

class MMData(Dataset):
    
    def __init__(self, train=True, transforms=None, 
                    flows = True, saved_flows=False,
                    dataset_path='../data', phase_fileformat='*.png',
                    labels_fileformat='*.png', mothermachine_data=False,
                    ):
        super(MMData, self).__init__()
        self.train = train
        self.test = not self.train
        self.phase_fileformat = phase_fileformat
        self.labels_fileformat = labels_fileformat
        self.transforms = transforms
        self.flows = flows
        self.saved_flows = saved_flows
        self.mothermachine_data = mothermachine_data
        
        # for Omnipose people data
        if self.train and (not self.mothermachine_data):
            phase_str = 'bacteria_train'
            if self.flows and self.saved_flows:
                labels_str = 'train_flows'
            else:
                labels_str = 'train_masks'
        elif self.test and (not self.mothermachine_data):
            phase_str = 'bacteria_test'
            if self.flows and self.saved_flows:
                labels_str = 'test_flows'
            else:
                labels_str = 'test_masks'

        # for our data
        if self.train and self.mothermachine_data:
            phase_str = 'phase_train'
            if self.flows and self.saved_flows:
                labels_str = 'flows_train'
            else:
                labels_str = 'masks_train'
        
        # for our data
        elif self.test and self.mothermachine_data:
            phase_str = 'phase_test'
            if self.flows and self.saved_flows:
                labels_str = 'flows_test'
            else:
                labels_str = 'masks_test'


        self.phase_dir = Path(dataset_path) / phase_str
        self.labels_dir = Path(dataset_path) / labels_str
        self.phase_filenames = list(self.phase_dir.glob(self.phase_fileformat))
        if not self.mothermachine_data:
            self.phase_filenames = sorted(self.phase_filenames, key=lambda x: int(x.stem.split('_')[0]))
        else:
            self.phase_fileformat = sorted(self.phase_filenames, key=lambda x: int(x.stem.split('_')[1]))
        #self.labels_filenames = list(self.labels_dir.glob(self.labels_fileformat))
        
    def __getitem__(self, idx):
        
        phase_filepath = self.phase_filenames[idx]

        if self.flows == True and self.saved_flows:
            labels_filename = phase_filepath.stem + '_flows' + self.labels_fileformat[1:]
        else:
            if self.mothermachine_data:
                labels_filename = phase_filepath.stem + self.labels_fileformat[1:]
            else:
                labels_filename = phase_filepath.stem + '_masks' + self.labels_fileformat[1:]

        # grab the phase image
        phase = io.imread(phase_filepath)

        # depending on options grab and create the labels image/imagestack
        if not self.flows:
            # no flows so, just grab the labels image from the directory
            labels = io.imread(self.labels_dir / labels_filename)
        else:
            if self.saved_flows:
                # just read the stack and put it in labels
                if self.labels_fileformat == "*.png"  or self.labels_fileformat == "*.tiff":
                    labels = io.imread(self.labels_dir / labels_filename)
                elif self.labels_fileformat == "*.npy":
                    labels = np.load(self.labels_dir / labels_filename)
            else:
                # construct flows by getting the masks and doing the flow calculations
                label_img = io.imread(self.labels_dir/ labels_filename)
                # calculate flows
                label_img, dists, heat, flows = labels_to_flows_cpu_omni(label_img)

                # Final_labels will be (5, H, W) in shape
                final_labels = np.concatenate((label_img[np.newaxis,:,:],
                                               dists[np.newaxis, :, :],
                                               flows,
                                               heat[np.newaxis, :, :]), axis=0).astype(np.float32)

                dist_bg = 5
                dist_t = final_labels[1]
                dist_t[dist_t == 0] = -5.0

                boundary = 5.0 * (final_labels[1] == 1)
                boundary[boundary == 0] = -5.0

                # add boundary to the label_stack 
                final_labels = np.concatenate((final_labels, boundary[np.newaxis,]))
                # add binary mask to the label_stack
                binary_mask = final_labels[0] > 0
                final_labels = np.concatenate((final_labels, binary_mask[np.newaxis,]))

                # add weights, an empty matrix that will be calculated during transformations
                
                bg_edt = edt.edt(final_labels[0] < 0.5, black_border=True)
                cutoff = 9
                weights = (gaussian(1 - np.clip(bg_edt, 0, cutoff)/cutoff, 1) + 0.5)
                #weights = np.zeros_like(final_labels[0])

                labels = np.concatenate((final_labels, weights[np.newaxis,]))
    
        # now we have both phase, labels, construct a sample
        sample = {
            'phase': phase,
            'labels': labels,
            'filename': str(phase_filepath)
        }
        if self.transforms:

            return self.transforms(sample)

        else:
            return sample
    
    def __len__(self):
       return len(self.phase_filenames)
    
    def plot(self, idx):
        datapoint = self.__getitem__(idx)

        if type(datapoint['phase']) == torch.Tensor:
            datapoint['phase'] = datapoint['phase'].numpy().squeeze(0)
            datapoint['labels'] = datapoint['labels'].numpy()

        if datapoint['labels'].shape[0] == 8:
            # do 9 plots
            fig, ax = plt.subplots(3, 3, figsize=(10, 8))
            labels_str= ['phase','labels', 'dists', 'flow-y', 'flow-x', 'heat', 'boundary', 'binary mask', 'weight']
            ax[0, 0].imshow(datapoint['phase'], cmap='gray')
            ax[0, 0].set_title(labels_str[0])
            for i in range(1, datapoint['labels'].shape[0] + 1):
                ax[i//3, i%3].imshow(datapoint['labels'][i-1])
                ax[i//3, i%3].set_title(labels_str[i])
            plt.suptitle(str(datapoint['filename']))
            plt.show()
                    
        elif datapoint['labels'].shape[0] == 1:
            fig, ax = plt.subplots(1, 2, figsize=(8, 4))
            labels_str = ['phase', 'labels']
            ax[0].imshow(datapoint['phase'], cmap='gray')
            ax[0].set_title(labels_str[0])
            ax[1].imshow(datapoint['labels'][0])
            ax[1].set_title(labels_str[1])
            plt.suptitle(str(datapoint['filename']))
            plt.show()
        else:
            fig, ax = plt.subplots(1, 2, figsize=(8, 4))
            labels_str = ['phase', 'labels']
            ax[0].imshow(datapoint['phase'], cmap='gray')
            ax[0].set_title(labels_str[0])
            ax[1].imshow(datapoint['labels'])
            ax[1].set_title(labels_str[1])
            plt.suptitle(str(datapoint['filename']))
            plt.show()
 



class MMDataTest(Dataset):

    def __init__(self, dataset_path='../data', transforms=None,
                    phase_fileformat='*.png', mothermachine_data=False):
        super(MMDataTest, self).__init__()

        if type(dataset_path) == pathlib.PosixPath:
            self.dataset_path = dataset_path
        elif type(dataset_path) == str:
            self.dataset_path = Path(dataset_path)

        self.mothermachine_data = mothermachine_data
        self.phase_fileformat = phase_fileformat
        self.transforms = transforms

        self.phase_filenames = list(self.dataset_path.glob(self.phase_fileformat))
        if self.mothermachine_data:
            self.phase_filenames = sorted(self.phase_filenames, key=lambda x: int(x.stem.split('_')[1]))
        else:
            self.phase_filenames = sorted(self.phase_filenames, key=lambda x: int(x.stem.split('_')[0]))

    def __len__(self):
        return len(self.phase_filenames)

    def __getitem__(self, idx):

        phase_filepath = self.phase_filenames[idx]

        # grab the phase image
        phase = io.imread(phase_filepath)
        H, W = phase.shape

        sample = {
            'phase': phase, 
            'filename': str(phase_filepath.resolve()),
            'raw_shape': (H, W)
        }

        if self.transforms:
            return self.transforms(sample)
        else:
            return sample

    def plot(self, idx):
        datapoint = self.__getitem__(idx)

        if type(datapoint['phase']) == torch.Tensor:
            datapoint['phase'] = datapoint['phase'].numpy().squeeze(0)

        fig, ax = plt.subplots(1, 1)
        ax.imshow(datapoint['phase'], cmap='gray')
        ax.set_title('Phase image')
        plt.suptitle(str(datapoint['filename']))
        plt.show()




# Use this for U-net training only, older format, with weighted loss function
class mmDataMultiSpecies(object):

    def __init__(self, trainingDataDir, species, transforms = None, datasetType='train', includeWeights=False):
        self.trainingDataDir = pathlib.Path(trainingDataDir)
        self.species = species
        self.datasetType = datasetType
        self.transforms = transforms
        self.includeWeights = includeWeights
        # construct all the file names  for all the species in the images, species dir will also have an
        # empty dir, containing empty channels, don't worry about this
        self.phaseDirs = []
        self.maskDirs = []
        self.weightDirs = []
        for dirName in self.species:
            self.phaseDirs.append(self.trainingDataDir.joinpath(dirName , 'phase_' + self.datasetType))
            self.maskDirs.append(self.trainingDataDir.joinpath(dirName, 'mask_' + self.datasetType))
            if self.includeWeights:
                self.weightDirs.append(self.trainingDataDir.joinpath(dirName, 'weights_' + self.datasetType))

        self.phaseFilenames = []
        self.maskFilenames = []
        self.weightFilenames = []

        for i, directory in enumerate(self.phaseDirs, 0):
            # grab filenames and add the corresponding mask to the file list
            filenames = [filename.name for filename in directory.glob('*.tif')]  
            for filename in filenames:
                # adding phaseFilenames
                self.phaseFilenames.append(directory.joinpath(filename))
                # adding maskFilenames
                self.maskFilenames.append(self.maskDirs[i].joinpath(filename))
                # adding weightFilenames
                if self.includeWeights:
                    self.weightFilenames.append(self.weightDirs[i].joinpath(filename))

    def __len__(self):
        return len(self.phaseFilenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()

        phaseImg = io.imread(self.phaseFilenames[idx])
        maskImg = io.imread(self.maskFilenames[idx])

        if self.includeWeights:
            weightImg = io.imread(self.weightFilenames[idx])
            weightFilename = self.weightFilenames[idx]
        else:
            weightImg = np.zeros(phaseImg.shape)
            weightFilename = None

        sample = {'phase': phaseImg, 'mask': maskImg, 'weights': weightImg,
                'phaseFilename': self.phaseFilenames[idx], 'maskFilename': self.maskFilenames[idx],
                'weightsFilename': weightFilename,
                 'filename': self.phaseFilenames[idx].name
                 }
        
        if self.transforms != None:
            sample = self.transforms(sample)

        return sample

    # This will plot the data point that goes into the training net
    def plotDataPoint(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()

        if type(idx) == list:
            print("Plotter only works with integer indices, can only plot one item at a time :( ")
            return
        else:
            fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
            data = self.__getitem__(idx)
            ax1.imshow(data['phase'].numpy().squeeze(0), cmap='gray')
            #ax1.set_xlabel(str(data['phaseFilename']))
            ax1.set_xlabel('Phase')
            #ax1.set_title('Phase')

            ax2.imshow(data['mask'].numpy().squeeze(0), cmap='gray')
            #ax2.set_xlabel(str(data['maskFilename']))
            ax2.set_xlabel('Binary mask')
            #ax2.set_title('mask')

            ax3.imshow(data['weights'].numpy().squeeze(0))
            #ax3.set_xlabel(str(data['weightFilename']))
            ax3.set_xlabel('Weight map')
            #ax3.set_title('weights')
            plt.show()


# Use this for U-net testing time only
class phaseTestDir(object):
    """
    Binding class that brings phase images in a directory together for batching and segmenting
    main Directory is going to be a positon directory and not the main main Dir containing positions
    """
    def __init__(self, phase_directory, transform = None, 
                phase_fileformat = '*.tiff'):

        if type(phase_directory) == str:
            self.phase_directory = pathlib.Path(phase_directory)
        else:
            self.phase_directory = phase_directory

        self.phase_fileformat = phase_fileformat
        self.transform = transform
        self.phase_filenames = sorted(list(self.phase_directory.glob(self.phase_fileformat)))
        self.n_images = len(self.phase_filenames)

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        phase_img_name = self.phase_filenames[idx]

        phase_img = io.imread(phase_img_name, as_gray = True)
        #phase_img = adjust_gamma(phase_img, gamma=1.2)
        phase_img = phase_img.astype('float32')
        H, W = phase_img.shape

#        if self.mean_std_normalize:
#            phase_mean = np.mean(phase_img)
#            phase_std = np.std(phase_img)
#
#            phase_img_normalized = (phase_img - phase_mean) / phase_std
#        else:
#            phase_img_normalized = phase_img
#
#        #phase_img[phase_img > 1.3] += 0.5
#
#        #phase_img = match_histograms(phase_img, self.imgref)
#        #phase_img = equalize_hist(phase_img)
#
#        #phase_img[phase_img > 1.0] += 0.5
#        
#        if self.add_noise:
#            rand_num = np.random.normal(0, 0.15, phase_img_normalized.shape)
#            phase_img_normalized = phase_img_normalized + rand_num
#
#       if self.flip:
#            phase_img_normalized = rotate(phase_img_normalized, angle = 180)
#
#        if self.pad_to != -1:
#            # pad to what ever 
#            pad_operation = padNumpyArray(pad_to=self.pad_to)
#            phase_img_normalized = pad_operation(phase_img_normalized)
#

        sample = {
            'phase': phase_img,
            'filename': phase_img_name,
            'raw_shape': (H, W)
        }
        
        if self.transform:
            sample = self.transform(sample)

        return {
            'phase': sample['phase'],
            'filename': str(sample['filename'].resolve()),
            'raw_shape': sample['raw_shape']
        }
        