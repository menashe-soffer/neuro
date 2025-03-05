import numpy as np
import pandas as pd


class my_montage_reader:

    def __init__(self, fname=None):

        if fname is not None:
            self.read_montage_file(fname=fname)

    def read_montage_file(self, fname):

        assert fname[-4:] == '.tsv'
        self.df = pd.read_csv(fname, delimiter='\t')


    def get_electrode_list_by_region(self, region_list, hemisphere_sel=None):

        # defualt hemisphere selectivity: no selectivity
        if hemisphere_sel is None:
            hemisphere_sel_list = ['both' for region in region_list]
        else:
            hemisphere_sel_list = hemisphere_sel

        electrode_list = dict()

        for i_region, region in enumerate(region_list):
            if hemisphere_sel_list[i_region] == 'both':
                electrode_list[region] = []
            else:
                if hemisphere_sel_list[i_region].find('L') > -1:
                    electrode_list[region + '-L'] = []
                if hemisphere_sel_list[i_region].find('R') > -1:
                    electrode_list[region + '-R'] = []

        for i in range(self.df.shape[0]):
            line = self.df.iloc[i]
            for i_region, region in enumerate(region_list):
                if ('ind.region' in line.keys()) and (region == line['ind.region']):

                    admit = False
                    if hemisphere_sel_list[i_region] == 'both':
                        admit, key = True, region
                    if hemisphere_sel_list[i_region].find(line['hemisphere']) > -1:
                        admit, key = True, region + '-' + line['hemisphere']
                    if admit:
                        electrode_list[key].append(dict({'name': line['name'], 'coords': np.array([line['x'], line['y'], line['z']]).reshape(3, 1), 'hemisphere': line['hemisphere']}))

        return electrode_list


if __name__ == '__main__':

    filename = 'E:/ds004789-download/sub-R1001P/ses-0/ieeg/sub-R1001P_ses-0_task-FR1_space-MNI152NLin6ASym_electrodes.tsv'
    montage = my_montage_reader(fname=filename)
    list = montage.get_electrode_list_by_region(['xxx', 'yyy', 'fusiform', 'lingual'], hemisphere_sel=['both', 'L', 'LR', 'both'])
    # now try a preliminary relevant list
    region_list = ['entorihinal', 'cuneus', 'fusiform', 'lateraloccipital', 'lingual', 'precuneus', 'superiorpariental']
    hemisphere_sel = ['LR', 'both', 'LR', 'LR', 'both', 'both', 'LR']
    list = montage.get_electrode_list_by_region(region_list=region_list, hemisphere_sel=hemisphere_sel)
    print(list)

