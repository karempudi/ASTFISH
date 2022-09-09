# Code for bundling and plotting all the growth rates according to species
import numpy as np
import glob
import matplotlib.pyplot as plt
import pickle
import seaborn as sb

np.seterr(divide='ignore', invalid='ignore')
def exp_growth_fit(x, a, b):
    return a * np.exp(-b * x)

class growth_rate_pickles(object):
    """
    A class that will take in all the pickles from all the positions
    given and cleanup all rates
    """

    def __init__(self, analysis_main_dir, species_names, species_titles, No_Ab_Positions=[],
        Ab_Positions=[], tracking_parameters=None, fileformat='*.pickle', n_frames=31,
        antibiotic_name='Nitro', antibiotic_concentration=4,
        color_map={
            'Klebsiella': 'r',
            'E.coli': 'b',
            'Pseudomonas': 'g',
            'E.cocci': 'm'
        }):
        
        self.analysis_main_dir = analysis_main_dir
        self.species_names = species_names
        self.species_titles = species_titles
        self.No_Ab_Positions = No_Ab_Positions
        self.Ab_Positions = Ab_Positions
        
        self.tracking_parameters = tracking_parameters
        self.fileformat = fileformat
        self.n_frames = n_frames

        self.antibiotic_name = antibiotic_name
        self.antibiotic_concentration = antibiotic_concentration
        self.color_map = color_map

        self.No_Ab_Directories_List = []
        self.Ab_Directories_List = []
        
        for position in self.No_Ab_Positions:
            dir_string = 'Pos' + str(position)
            self.No_Ab_Directories_List.append(self.analysis_main_dir / dir_string / self.tracking_parameters['write_dir_names']['growth_rates'])

        for position in self.Ab_Positions:
            dir_string = 'Pos' + str(position) 
            self.Ab_Directories_List.append(self.analysis_main_dir / dir_string / self.tracking_parameters['write_dir_names']['growth_rates'])

        self.No_Ab_Pickle_FilesList = []
        self.Ab_Pickle_FilesList = []

        for directory in self.No_Ab_Directories_List:
            self.No_Ab_Pickle_FilesList.extend(list(directory.glob(self.fileformat)))
        
        for directory in self.Ab_Directories_List:
            self.Ab_Pickle_FilesList.extend(list(directory.glob(self.fileformat)))

        # puts together all the pickle files and then adds up the species dictionaries
        self.construct_growth_rates()
        self.clean_growth_rates()


    def __len__(self):
        return len(self.No_Ab_Pickle_FilesList) + len(self.Ab_Pickle_FilesList)

    def construct_growth_rates(self):
        self.No_Ab_Species_GrowthRates = {}
        self.Ab_Species_GrowthRates = {}

        self.No_Ab_Pooled_GrowthRates = []
        self.Ab_Pooled_GrowthRates = []

        self.No_Ab_Counts = {}
        self.Ab_Counts = {}

        for species in self.species_names:
            self.No_Ab_Species_GrowthRates[species] = []
            self.Ab_Species_GrowthRates[species] = []

            self.No_Ab_Counts[species] = 0
            self.Ab_Counts[species] = 0
        
        for filename in self.No_Ab_Pickle_FilesList:
            with open(filename, 'rb') as filehandle:
                data = pickle.load(filehandle)

            for key, value in data.items():
                if value != []:
                    self.No_Ab_Counts[key] += 1
                
                self.No_Ab_Species_GrowthRates[key] += value
                self.No_Ab_Pooled_GrowthRates += value

        self.No_Ab_Pooled_GrowthRates = np.array(self.No_Ab_Pooled_GrowthRates)

        for filename in self.Ab_Pickle_FilesList:
            with open(filename, 'rb') as filehandle:
                data = pickle.load(filehandle)

            for key, value in data.items():
                if value != []:
                    self.Ab_Counts[key] += 1

                self.Ab_Species_GrowthRates[key] += value
                self.Ab_Pooled_GrowthRates += value

        self.Ab_Pooled_GrowthRates = np.array(self.Ab_Pooled_GrowthRates)

    def __getitem__(self, idx):
        pass

    def clean_growth_rates(self, frame_rate=2):
        self.No_Ab_Clean_GrowthRates = {}
        self.Ab_Clean_GrowthRates = {}

        for species in self.species_names:
            self.No_Ab_Clean_GrowthRates[species] = self.mean_std_counts(species, frame_rate=frame_rate, ab = False)
            self.Ab_Clean_GrowthRates[species] = self.mean_std_counts(species, frame_rate=frame_rate, ab = True)
        
        No_Ab_pool_counts = np.zeros(shape=(self.n_frames,))
        Ab_pool_counts = np.zeros(shape=(self.n_frames,))
        No_Ab_pool_growth = np.zeros(shape=(self.n_frames,))
        Ab_pool_growth = np.zeros(shape=(self.n_frames,))
        No_Ab_pool_growth_dev = np.zeros(shape=(self.n_frames,))
        Ab_pool_growth_dev = np.zeros(shape=(self.n_frames,))

        for i in range(self.n_frames):
            No_Ab_column = self.No_Ab_Pooled_GrowthRates[:, i]
            Ab_column = self.Ab_Pooled_GrowthRates[:, i]

            if len(No_Ab_column[No_Ab_column != -1]) == 0:
                pass
            else:
                No_Ab_pool_growth[i] = np.mean(No_Ab_column[np.logical_and(No_Ab_column != -1, No_Ab_column >= 0)] / frame_rate)
                No_Ab_pool_counts[i] = np.sum(np.logical_and(No_Ab_column != -1, No_Ab_column >= 0))
                No_Ab_pool_growth_dev[i] = np.std(No_Ab_column[np.logical_and(No_Ab_column != -1, No_Ab_column >= 0)] / frame_rate)

            if len(Ab_column[Ab_column != -1]) == 0:
                pass
            else:
                Ab_pool_growth[i] = np.mean(Ab_column[np.logical_and(Ab_column != -1, Ab_column >= 0)] / frame_rate)
                Ab_pool_counts[i] = np.sum(np.logical_and(Ab_column != -1, Ab_column >= 0))
                Ab_pool_growth_dev[i] = np.std(Ab_column[np.logical_and(Ab_column != -1, Ab_column >= 0)] / frame_rate)
            
        self.No_Ab_Clean_Pooled_GrowthRates = (No_Ab_pool_growth, No_Ab_pool_growth_dev, No_Ab_pool_counts)
        self.Ab_Clean_Pooled_GrowthRates = (Ab_pool_growth, Ab_pool_growth_dev, Ab_pool_counts)


    def mean_std_counts(self, species, frame_rate = 2, ab = False):
        if ab == False:
            species_growth_rates = np.array(self.No_Ab_Species_GrowthRates[species])
            counts_t_species = np.zeros(shape=(self.n_frames,))
            growth_t_species = np.zeros(shape=(self.n_frames,)) 
            growth_dev_t_species = np.zeros(shape=(self.n_frames,))

            if len(species_growth_rates) == 0:
                return growth_t_species, growth_dev_t_species, counts_t_species

            for i in range(self.n_frames):
                column = species_growth_rates[:, i]
                if len(column[column != -1]) == 0:
                    pass
                else:
                    growth_t_species[i] = np.mean(column[np.logical_and(column != -1, column >= 0)] / frame_rate)
                    counts_t_species[i] = np.sum(np.logical_and(column != -1, column >= 0))
                    growth_dev_t_species[i] = np.std(column[np.logical_and(column != -1, column >= 0)] / frame_rate)

            return growth_t_species, growth_dev_t_species, counts_t_species

        elif ab == True:
            species_growth_rates = np.array(self.Ab_Species_GrowthRates[species])
            counts_t_species = np.zeros(shape=(self.n_frames,))
            growth_t_species = np.zeros(shape=(self.n_frames,)) 
            growth_dev_t_species = np.zeros(shape=(self.n_frames,))

            if len(species_growth_rates) == 0:
                return growth_t_species, growth_dev_t_species, counts_t_species

            for i in range(self.n_frames):
                column = species_growth_rates[:, i]
                if len(column[column != -1]) == 0:
                    pass
                else:
                    growth_t_species[i] = np.mean(column[np.logical_and(column != -1, column >= 0)] / frame_rate)
                    counts_t_species[i] = np.sum(np.logical_and(column != -1, column >= 0))
                    growth_dev_t_species[i] = np.std(column[np.logical_and(column != -1, column >= 0)] / frame_rate)

            return growth_t_species, growth_dev_t_species, counts_t_species

    def get_growth_rates(self, species):
        if species not in self.species_names:
            return None

        species_no_ab = self.No_Ab_Clean_GrowthRates[species]
        species_ab = self.Ab_Clean_GrowthRates[species]

        normalized_growth_rates = species_ab[0]/species_no_ab[0]
        species_err_no_ab = species_no_ab[1]/species_no_ab[0]/np.sqrt(species_no_ab[2])
        species_err_ab = species_ab[1]/species_no_ab[0]/np.sqrt(species_ab[2])

        return (normalized_growth_rates, species_err_no_ab, species_err_ab)

    def plot_species_wise_and_pooled(self, color_scheme, species_full_name = { 'Klebsiella': "K. pneumoniae", 
        "E.coli": "E.coli", "Pseudomonas": "P.aeruginosa", "E.cocci": "E.faecalis"},
        ignore = []):
        sb.set_style("white")
        fig, ax = plt.subplots(nrows=1, ncols=1)

        for i in range(len(self.species_names)):
        
            if self.species_names[i] in ignore:
                continue
            species_noab = self.No_Ab_Clean_GrowthRates[self.species_names[i]]
            species_ab = self.Ab_Clean_GrowthRates[self.species_names[i]]
            normalized_growth_rates = species_ab[0]/species_noab[0]
            species_err_no_ab = species_noab[1]/species_noab[0]/np.sqrt(species_noab[2])
            species_err_ab = species_ab[1]/species_noab[0]/np.sqrt(species_ab[2])

            ax.plot(range(0, 2*self.n_frames, 2), normalized_growth_rates, color=color_scheme[self.species_names[i]], 
                    label=species_full_name[self.species_names[i]]+ ' Treatment')
            ax.fill_between(range(0, 2 * self.n_frames, 2), normalized_growth_rates - species_err_ab,
                            normalized_growth_rates + species_err_ab,
                            alpha = 0.4, color= color_scheme[self.species_names[i]], linestyle='--', linewidth=2)
        
         
        # plot some pooled curve here to see if you can differentiate
    
        noAb_pool = self.No_Ab_Clean_Pooled_GrowthRates
        Ab_pool = self.Ab_Clean_Pooled_GrowthRates
        normalized_pool = Ab_pool[0]/noAb_pool[0]
        noAb_pool_err = noAb_pool[1]/noAb_pool[0]/np.sqrt(noAb_pool[2])
        Ab_pool_err = Ab_pool[1]/noAb_pool[0]/np.sqrt(Ab_pool[2])

        # Normalized growth rates no antibiotic
        ax.plot(range(0, 2*self.n_frames, 2), [1] * self.n_frames, 'k', label='No Species Id Reference')

        # standard error of the normalized values
        ax.fill_between(range(0, 2 * self.n_frames, 2), [1]* self.n_frames - noAb_pool_err, 
                    [1] * self.n_frames + noAb_pool_err, alpha=0.4, color='k', linestyle='--', linewidth=2)

        # standard deviation of the normalized values
        #ax[1, 1].fill_between(range(0, 2 * self.nFrames, 2), [1]* self.nFrames - noAb_pool[1]/noAb_pool[0], 
        #            [1] * self.nFrames + noAb_pool[1]/noAb_pool[0], alpha=0.2, color='b')

        # normalized growth rates - with antibiotic
        ax.plot(range(0, 2*self.n_frames, 2), normalized_pool,
                 'k:', label='No species Id Treatment')
        ax.fill_between(range(0, 2 * self.n_frames, 2), normalized_pool - Ab_pool_err, 
                    normalized_pool + Ab_pool_err, alpha=0.4, color='k', linestyle=':', linewidth=2)
        #ax[1, 1].fill_between(range(0, 2 * self.nFrames, 2), normalized_pool - Ab_pool[1]/noAb_pool[0], 
        #            normalized_pool + Ab_pool[1]/noAb_pool[0], alpha=0.2, color='r')
        #ax.set_ylim([0, 1.4])
        #ax.set_ylabel("Growth Rate (normalized)")
        #ax.set_xlabel("Time(min)")
        #ax.legend(loc='lower left')


        
        ax.set_xlim([8, 2 * self.n_frames - 2])
        ax.set_ylabel("Growth Rate (normalized)")
        ax.set_title(f"{self.antibiotic_name} {self.antibiotic_concentration}" + r'$\mu g/ml$ ')
        plt.xticks(fontsize=12, weight='bold')
        ax.set_ylim([0, 1.4])
        plt.yticks(fontsize=12, weight='bold')
        ax.legend(loc='best',fontsize='large',framealpha=0.3)
        ax.set_xlabel("Time(min)")


    # ignore is a list of species you can ignore while plotting
    def plot_all_figures(self, std_err=True, std_dev=True, ignore = [], ylim=1.6, 
            speciesFullName = { 'Klebsiella': "K. pneumoniae", 
                                "E.coli": "E.coli",
                                "Pseudomonas": "P.aeruginosa",
                                "E.cocci": "E.faecalis"}):
        nrows = 2
        ncols = 4
        sb.set_style("white")
        fig, ax = plt.subplots(nrows=2, ncols=4)
        for i in range(len(self.species_names)):
            # species_noab[0] -- mean growth rates
            # species_noab[1] -- std dev growth rates
            # species_noab[2] -- no of tracks at any timepoint
            if self.species_names[i] in ignore:
                continue
            species_noab = self.No_Ab_Clean_GrowthRates[self.species_names[i]]
            species_ab = self.Ab_Clean_GrowthRates[self.species_names[i]]
            normalized_growth_rates = species_ab[0]/species_noab[0]
            species_err_no_ab = species_noab[1]/species_noab[0]/np.sqrt(species_noab[2])
            species_err_ab = species_ab[1]/species_noab[0]/np.sqrt(species_ab[2])

            # plotting std_err and std_dev of normalized growth rates
            
            # Normalized growth rates no antibiotic
            ax[0, i].plot(range(0, 2*self.n_frames, 2), [1] * self.n_frames, 'b-', label='Reference')

            # standard error of the normalized values

            ax[0, i].fill_between(range(0, 2 * self.n_frames, 2), [1]* self.n_frames - species_err_no_ab, 
                        [1] * self.n_frames + species_err_no_ab, alpha=0.4, color='b', linestyle='--', linewidth=2)

            # standard deviation of the normalized values
            #ax[0, i].fill_between(range(0, 2 * self.nFrames, 2), [1]* self.nFrames - species_noab[1]/species_noab[0], 
            #            [1] * self.nFrames + species_noab[1]/species_noab[0], alpha=0.2, color='b')

            # normalized growth rates - with antibiotic
            ax[0, i].plot(range(0, 2*self.n_frames, 2), normalized_growth_rates, 'r-', label='Treatment')
            ax[0, i].fill_between(range(0, 2 * self.n_frames, 2), normalized_growth_rates - species_err_ab, 
                        normalized_growth_rates + species_err_ab, alpha=0.4, color='r', linestyle='--', linewidth=2)
            #ax[0, i].fill_between(range(0, 2 * self.nFrames, 2), normalized_growth_rates - species_ab[1]/species_noab[0], 
            #            normalized_growth_rates + species_ab[1]/species_noab[0], alpha=0.2, color='r')
            ax[0, i].set_xlim([8, 2*self.n_frames - 2])
            ax[0, i].set_ylim([0, 1.4])
            ax[0, i].set_title(f"{self.antibiotic_name} {self.antibiotic_concentration}" + r'$\mu g/ml$ ' + f"{speciesFullName[self.species_names[i]]}")
            ax[0, i].set_ylabel("Growth Rate (normalized)")
            ax[0, i].set_xlabel("Time(min)")
            ax[0, i].legend(loc='lower left')


        # plot relative curves based on species

        ax[1, 0].plot(range(0, 2 * self.n_frames, 2), [1] * self.n_frames, color='k', label='Reference')
        for i in range(len(self.species_names)):
            if self.species_names[i] in ignore:
                continue
            species_noab = self.No_Ab_Clean_GrowthRates[self.species_names[i]]
            species_ab = self.Ab_Clean_GrowthRates[self.species_names[i]]
            normalized_growth_rates = species_ab[0]/species_noab[0]
            species_err_ab = species_ab[1]/species_noab[0]/np.sqrt(species_ab[2])

            ax[1, 0].plot(range(0, 2*self.n_frames, 2), normalized_growth_rates, color=self.color_map[self.species_names[i]], label=speciesFullName[self.species_names[i]])
            ax[1, 0].fill_between(range(0, 2 * self.n_frames, 2), normalized_growth_rates - species_err_ab, normalized_growth_rates + species_err_ab,
                                  alpha = 0.4, color= self.color_map[self.species_names[i]], linestyle='--', linewidth=2)
            
        ax[1, 0].set_xlim([8, 2 * self.n_frames - 2])
        ax[1, 0].set_ylabel("Growth Rate (normalized)")
        ax[1, 0].set_title(f"{self.antibiotic_name} {self.antibiotic_concentration}" + r'$\mu g/ml$ ' + " All species")
        ax[1, 0].set_ylim([0, 1.4])
        ax[1, 0].legend(loc='best',fontsize='medium',framealpha=0.3)
        ax[1, 0].set_xlabel("Time(min)")
        




        # plot number of cells available at each time point
        for i in range(len(self.species_names)):
            if self.species_names[i] in ignore:
                continue
            
            species_noab = self.No_Ab_Clean_GrowthRates[self.species_names[i]]
            species_ab = self.Ab_Clean_GrowthRates[self.species_names[i]]

            ax[1, 2].plot(range(0, 2 *self.n_frames, 2), species_noab[2], color=self.color_map[self.species_names[i]], label=speciesFullName[self.species_names[i]]+ " Reference")
            ax[1, 2].plot(range(0, 2 * self.n_frames, 2), species_ab[2], color=self.color_map[self.species_names[i]], linestyle='--', label=speciesFullName[self.species_names[i]]+ " Treatment")
        
        ax[1, 2].legend(loc='upper left', fontsize='x-small')
        ax[1, 2].set_xlim([8, 2*self.n_frames -2])
        ax[1, 2].set_xlabel("Time(min)")
        ax[1, 2].set_ylabel("Number of cells")


        # plot some pooled curve here to see if you can differentiate
    
        noAb_pool = self.No_Ab_Clean_Pooled_GrowthRates
        Ab_pool = self.Ab_Clean_Pooled_GrowthRates
        normalized_pool = Ab_pool[0]/noAb_pool[0]
        noAb_pool_err = noAb_pool[1]/noAb_pool[0]/np.sqrt(noAb_pool[2])
        Ab_pool_err = Ab_pool[1]/noAb_pool[0]/np.sqrt(Ab_pool[2])


        # Normalized growth rates no antibiotic
        ax[1, 1].plot(range(0, 2*self.n_frames, 2), [1] * self.n_frames, 'b-', label='Reference')

        # standard error of the normalized values

        ax[1, 1].fill_between(range(0, 2 * self.n_frames, 2), [1]* self.n_frames - noAb_pool_err, 
                    [1] * self.n_frames + noAb_pool_err, alpha=0.4, color='b', linestyle='--', linewidth=2)

        # standard deviation of the normalized values
        #ax[1, 1].fill_between(range(0, 2 * self.nFrames, 2), [1]* self.nFrames - noAb_pool[1]/noAb_pool[0], 
        #            [1] * self.nFrames + noAb_pool[1]/noAb_pool[0], alpha=0.2, color='b')

        # normalized growth rates - with antibiotic
        #print(normalized_pool)
        ax[1, 1].plot(range(0, 2*self.n_frames, 2), normalized_pool, 'r-', label='Treatment')
        ax[1, 1].fill_between(range(0, 2 * self.n_frames, 2), normalized_pool - Ab_pool_err, 
                    normalized_pool + Ab_pool_err, alpha=0.4, color='r', linestyle='--', linewidth=2)
        #ax[1, 1].fill_between(range(0, 2 * self.nFrames, 2), normalized_pool - Ab_pool[1]/noAb_pool[0], 
        #            normalized_pool + Ab_pool[1]/noAb_pool[0], alpha=0.2, color='r')
        ax[1, 1].set_xlim([8, 2*self.n_frames - 2])
        #ax[0, i].set_ylim([0, 5])
        ax[1, 1].set_title(f"{self.antibiotic_name} {self.antibiotic_concentration}" + r'$\mu g/ml$ ' + "Pooled. No species ID")
        ax[1, 1].set_ylim([0, 1.4])
        ax[1, 1].set_ylabel("Growth Rate (normalized)")
        ax[1, 1].set_xlabel("Time(min)")
        ax[1, 1].legend(loc='lower left')



        # plot channel counts to see if the species were loaded equally on both sides
        labels = [speciesFullName[species] for species in self.species_names]
        noAbCounts = []
        AbCounts = []
        for species in self.species_names:
            noAbCounts.append(self.No_Ab_Counts[species])
            AbCounts.append(self.Ab_Counts[species])
        
        x = np.arange(len(labels))
        width = 0.35
        ax[1, 3].bar(x - width/2, noAbCounts, width, label='Reference', color='b')
        ax[1, 3].bar(x + width/2, AbCounts, width, label='Treatment', color='r')
        ax[1, 3].set_ylabel("Number of channels")
        ax[1, 3].set_title("Species vs Number of channels")
        ax[1, 3].set_xticks(x)
        ax[1, 3].set_xticklabels(labels)
        ax[1, 3].legend()
        print(labels)



        plt.subplots_adjust(hspace=0.207, wspace=0.245, top=0.960, bottom=0.060, left=0.042, right=0.986)
