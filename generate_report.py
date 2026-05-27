import os
from pptx import Presentation
from pptx.util import Inches
from pdf2image import convert_from_path
from pptx.dml.color import RGBColor
from pptx.util import Pt # If you also want to change the font size in points
import pickle

from paths_and_constants import *



ALL_REGIONS = ['bankssts', 'caudalanteriorcingulate', 'caudalmiddlefrontal', 'cuneus', 'entorhinal', 'frontalpole', 'fusiform', 
               'inferiorparietal', 'inferiortemporal', 'insula', 'isthmuscingulate', 'lateraloccipital', 'lateralorbitofrontal',
               'lingual', 'medialorbitofrontal', 'middletemporal', 'mix', 'nan', 'paracentral', 'parahippocampal', 'parsopercularis',
               'parsorbitalis', 'parstriangularis', 'pericalcarine', 'postcentral', 'posteriorcingulate', 'precentral', 'precuneus',
               'rostralanteriorcingulate', 'rostralmiddlefrontal', 'superiorfrontal', 'superiorparietal', 'superiortemporal',
               'supramarginal', 'temporalpole', 'transversetemporal']

RESPONSIVE_REGIONS = ['cuneus', 'pericalcarine', 'postcentral', 'precentral', 'lingual', 'superiorparietal', 
                      'inferiortemporal', 'middletemporal', 'fusiform', 'lateraloccipital']

EXTENSION_REGIONS = ['bankssts', 'caudalmiddlefrontal', 'inferiorparietal', 'insula', 'lateralorbitofrontal', 'lingual',
                     'parsopercularis', 'parstriangularis', 'precuneus', 'rostralmiddlefrontal', 'superiorfrontal',
                     'superiorparietal', 'superiortemporal', 'superiortemporal']



import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


data = dict({'ALL': dict(), 'HIGH_RESP': dict()})
for selectby in ['all', 'odd', 'even']:
    for calc in ['all', 'odd', 'even']:
        for mode in ['ALL', 'HIGH_RESP']:
            fname = os.path.join(FIG_FOLDER, 'selectby_{}_calc_{}'.format(selectby, calc), 'CNTDWN_USE_{}_SPLIT_ALL'.format(mode), 'data_for_figures_11')
            if os.path.isfile(fname):
                with open(fname, 'rb') as f:
                    data[mode]['{}_{}'.format(selectby, calc)] = pickle.load(f)
#

def my_new_slide(doc, slide_title=''):
     
    slide = doc.slides.add_slide(doc.slide_layouts[5]) # title only
    title_box = slide.placeholders[0]
    tf = title_box.text_frame
    tf.text = slide_title 
    p = tf.paragraphs[0]
    p.text = slide_title
    p.font.name = 'Arial'          # Change font family (e.g., Arial, Calibri)
    p.font.size = Pt(24)           # Change font size
    p.font.bold = True             # Make it bold
    p.font.color.rgb = RGBColor(0x5, 0x40, 0xd0) # Dark Gray (Hex: #333333)

    return slide


doc = Presentation()

for region in RESPONSIVE_REGIONS + EXTENSION_REGIONS:

    # add first slide
    slide = my_new_slide(doc, slide_title=region)

    # create consistancy statistics and put in left part of the slide
    map_all = data['HIGH_RESP']['all_odd']['PSTH']['slct_masks'][region]
    map_odd = data['HIGH_RESP']['odd_odd']['PSTH']['slct_masks'][region]
    map_even = data['HIGH_RESP']['even_odd']['PSTH']['slct_masks'][region]
    select_mat = np.concatenate((map_all[np.newaxis, :], map_odd[np.newaxis, :], map_even[np.newaxis, :]), axis=0)
    # num_cntcts = select_mat.shape[1]
    # max_line_sz = 30
    # max_subplots = 6
    # num_subplots = np.ceil(num_cntcts / max_line_sz)
    # num_plots = 
    cmat = np.zeros((3,3), dtype=int)
    for i1 in range(3):
        for i2 in range(3):
            cmat[i1, i2] = (select_mat[i1] * select_mat[i2]).sum()
    fig, ax = plt.subplots(1, 1)
    sns.heatmap(cmat, ax=ax, xticklabels=['all_epochs', 'odd_epoch_subset', 'even_epoch_subset'], 
                yticklabels=['all_epochs', 'odd_epoch\nsubset', 'even_epoch\nsubset'], 
                square=True, cbar=False, annot=True, annot_kws={"size": 18}, fmt="d", vmin=0)
    fig.suptitle('{} , ({} contacts)'.format(region, select_mat.shape[1]))
    fig.savefig(os.path.join(TEMP_FOLDER, 'temp_fig'))
    #
    slide.shapes.add_picture(os.path.join(TEMP_FOLDER, 'temp_fig.png'), Inches(0), Inches(2), width=Inches(5))

    # create baseline psth and put in right part of the slide
    fig, ax = plt.subplots(1, 1)
    boundary_sec = data['HIGH_RESP']['even_odd']['PSTH']['boundary_sec']
    tscale = (boundary_sec[:-1] + boundary_sec[1:]) / 2
    psth = data['ALL']['all_all']['PSTH']['psth_set'][region][0][0]
    sem = data['ALL']['all_all']['PSTH']['psth_set'][region][0][1]
    line1, = ax.plot(tscale, np.log(np.maximum(psth, 1e-9)))
    ax.fill_between(tscale, np.log(np.maximum(psth - sem, 1e-6)),
                    np.log(np.maximum(psth + sem, 1e-6)), color=line1.get_color(), alpha=0.2)
    ax.set_ylim((-0.1, 0.3))
    ax.grid(True)
    fig.suptitle('{}  -  PSTH without contact selection\n({} contacts)'.format(region, select_mat.shape[1]))
    fig.savefig(os.path.join(TEMP_FOLDER, 'temp_fig.png'))
    #
    slide.shapes.add_picture(os.path.join(TEMP_FOLDER, 'temp_fig.png'), Inches(5), Inches(2), width=Inches(5))
    
    # create new slide for contact selection ("high response") psth grid
    slide = my_new_slide(doc, slide_title=r'{} - PSTH with contact selection'.format(region))

    # create "high response"" psth grid and put in slide
    fig, ax = plt.subplots(3, 3, figsize=(8, 5))
    for i_row, row_title in enumerate(['select_using_\nall_epochs', 'select_using_\nodd_epochs', 'select_using_\neven_epochs']):
        #ax[i_row, 0].yaxis.set_label_position("right")
        ax[i_row, 0].set_ylabel(row_title, labelpad=4 + (i_row > 0) * 24)
        #ax[i_row, 0].yaxis.set_label_position("right")
    for i_col, col_title in enumerate(['psth - all epochs', 'psth - odd epochs', 'psth - even epochs']):
        ax[0, i_col].set_title(col_title)
    for i_row, selectby in enumerate(['all', 'odd', 'even']):
        for i_col, calc in enumerate(['all', 'odd', 'even']):
            # boundary_sec = data['HIGH_RESP']['even_odd']['PSTH']['boundary_sec']
            # tscale = (boundary_sec[:-1] + boundary_sec[1:]) / 2
            try:
                selectby_calc = '{}_{}'.format(selectby, calc)
                psth = data['HIGH_RESP'][selectby_calc]['PSTH']['psth_set'][region][0][0]
                sem = data['HIGH_RESP'][selectby_calc]['PSTH']['psth_set'][region][0][1]
                line1, = ax[i_row, i_col].plot(tscale, np.log(np.maximum(psth, 1e-9)))
                ax[i_row, i_col].fill_between(tscale, np.log(np.maximum(psth - sem, 1e-6)), 
                                              np.log(np.maximum(psth + sem, 1e-6)), color=line1.get_color(), alpha=0.2)
                ax[i_row, i_col].set_ylim((-0.1, 0.3))
                ax[i_row, i_col].grid(True)
            except:
                #ax[i_row, i_col].axis(False)
                ax[i_row, i_col].set_xticks([])
                ax[i_row, i_col].set_yticks([])
    fig.savefig(os.path.join(TEMP_FOLDER, 'temp_fig'))
    #
    slide.shapes.add_picture(os.path.join(TEMP_FOLDER, 'temp_fig.png'), Inches(1), Inches(1.5), width=Inches(8), height=Inches(5))
    #
    #doc.save(os.path.join(TEMP_FOLDER, '27-5-26.pptx'))
    print('created 2 slides for', region)
    plt.close('all')
doc.save(os.path.join(TEMP_FOLDER, '27-5-26.pptx'))
    



assert False

doc = Presentation()

for region in ALL_REGIONS:
    for num_sessions in [1, 2, 3]:
        for cnct_mode in ['CNTCTS', 'EPOCHS'][:1]:
            for hr_mode in ['old_hr', 'new_hr', 'new_hr_v2', 'new_hr_v3']:
                if num_sessions == 1:
                    if cnct_mode == 'EPOCHS':
                        continue
                    folder_name = os.path.join(FIG_FOLDER, 'sess_1_{}'.format(hr_mode))
                    slide_title = '{} single session, {})'.format(region, hr_mode)
                else:
                    folder_name = os.path.join(FIG_FOLDER, 'sess_{}_{}'.format(num_sessions, hr_mode))
                    slide_title = '{}   {} sessions,  {}'.format(region, num_sessions, hr_mode)
                #
                print(folder_name, slide_title)
                #
                all_fname = os.path.join(folder_name, 'CNTDWN_USE_ALL_SPLIT_ALL', 'PSTH  ({}).pdf'.format(region))
                high_resp_fname = os.path.join(folder_name, 'CNTDWN_USE_HIGH_RESP_SPLIT_ALL', 'PSTH  ({}).pdf'.format(region))
                if os.path.isfile(all_fname):
                    # Add a blank slide
                    slide = doc.slides.add_slide(doc.slide_layouts[5]) # title only
                    #slide.placeholders[0].text = slide_title

                    # Access the text frame of the title placeholder
                    title_box = slide.placeholders[0]
                    tf = title_box.text_frame
                    
                    # Clear any default text, then get the first paragraph
                    tf.text = slide_title 
                    p = tf.paragraphs[0]
                    p.text = slide_title
                    
                    # --- Modify Font Style and Color ---
                    p.font.name = 'Arial'          # Change font family (e.g., Arial, Calibri)
                    p.font.size = Pt(24)           # Change font size
                    p.font.bold = True             # Make it bold
                    p.font.color.rgb = RGBColor(0x5, 0x40, 0xd0) # Dark Gray (Hex: #333333)

                    # convert pdf to png
                    image = convert_from_path(all_fname)
                    tmp_img_fname = all_fname.replace('.pdf', '.jpg')
                    image[0].save(tmp_img_fname, 'JPEG', quality=5)
                    # put png in left side
                    slide.shapes.add_picture(tmp_img_fname, Inches(0), Inches(2.5), width=Inches(5))
                    os.remove(tmp_img_fname)
                    if os.path.isfile(high_resp_fname):
                        # convert pdf to png 
                        image = convert_from_path(high_resp_fname)
                        tmp_img_fname = high_resp_fname.replace('.pdf', '.jpg')
                        image[0].save(tmp_img_fname, 'JPEG', quality=5)
                        # put png in right side
                        slide.shapes.add_picture(tmp_img_fname, Inches(5), Inches(2.5), width=Inches(5))
                        os.remove(tmp_img_fname)
#
doc.save(os.path.join(FIG_FOLDER, '24-5-26.pptx'))

