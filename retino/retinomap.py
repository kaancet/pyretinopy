import numpy as np
import os
import tifffile as tf
import cv2
from wfield import *
import scipy.ndimage as ni
import skimage.morphology as sm


class RetinoMaps:
    def __init__(self):
        self.main_path = None
        self.animalid = None
        self.expdate = None
        self.phases = None
        self.signmaps = None
        self.patchmaps = None
        self.save_path = '/Users/kaan/src/temp/retino'

    def init_RetinoMap(self, exp_path, params={'signThresh': 0.3,
                                               'areaThresh': 100,
                                               'openIter': 3
                                                 }):
        self.params = params
        self.main_path = exp_path
        self.get_metadata(exp_path)
        self.read_wfield_experiment(self.main_path)
        self.combine_stims(method='max')
        self.combine_stims(method='avg')
        self.phases = self.get_phasemap()
        self.signmaps = self.get_signmap()
        self.patchmaps = self.get_patchmap()

    def get_metadata(self,exp_path):
        # parse the main path to extract metadata
        tempsep = exp_path.split(sep='_')
        self.animalid = tempsep[1]
        self.expdate = tempsep[0].split(sep='/')[-1]

    def get_surfaceFOV(self, p):
        temp = [i for i in os.listdir(p) if 'DS' not in i]
        surfaceimg = tf.imread(os.path.join(p, temp[0]))
        if surfaceimg.shape[0] != 502 and surfaceimg.shape[1] != 501:
            surfaceimg = cv2.resize(surfaceimg, (502, 501), interpolation=cv2.INTER_AREA)
        return surfaceimg

    def combine(self, mov1, mov2, method):
        if mov1.shape != mov2.shape:
            raise IOError('The dimensions of the two movies should be the same')
        combined = np.empty_like(mov1[:, :, :])
        tdim = mov1.shape[0]
        for t in range(tdim):
            temp = np.dstack((mov1[t, :, :], mov2[t, :, :]))
            if method == 'max':
                tempmax = np.amax(temp, axis=2)
                combined[t, :, :] = tempmax
            elif method == 'avg':
                tempavg = np.mean(temp, axis=2)
                combined[t, :, :] = tempavg
        return combined

    def find_patches(self, smap, signthresh, openIter):
        pmap = np.zeros(smap.shape)
        pmap[smap >= signthresh] = 1
        pmap[smap <= -1 * signthresh] = 1
        pmap[(smap < signthresh) & (smap > -1 * signthresh)] = 0
        # open areas with the default structure
        # a custom large structure can be used to get rid of small areas
        pmap = ni.binary_opening(np.abs(pmap), iterations=openIter).astype(np.int)
        patchareas = sm.convex_hull_image(pmap)
        # define borders
        borders = np.multiply(-1 * (pmap - 1), patchareas)
        borders = sm.skeletonize(borders)
        # dilate borders
        patchBorder = ni.binary_dilation(borders, iterations=1).astype(np.int)
        # patches with dilated borders
        newpatch = np.multiply(-1 * (patchBorder - 1), patchareas)
        labeledpatches, patchnum = ni.label(newpatch)
        return pmap, labeledpatches, patchnum

    def read_wfield_experiment(self,exp_path):
        '''
        Reads averaged stimulus responses for lowSF and highSF, and the surface image in a given experiment folder
        '''

        mov_dict = {}
        surface = None
        # iterates in different runs
        for f in os.listdir(exp_path):

            if f == 'surface':
                surface = self.get_surfaceFOV(os.path.join(self.main_path, f))
            elif '.DS_Store' in f:
                continue
            else:
                path = os.path.join(exp_path, f, 'stimaverages_cam3')
                movlist = os.listdir(path)
                movlist.sort()
                # iterates in different stims
                for mov in movlist:
                    # get the stimulus number
                    stimno = list(filter(str.isdigit, mov))[0]
                    print('reading {0} in {1}'.format(mov, path))
                    temp = tf.imread(os.path.join(path, mov))  # read the stim
                    if int(stimno) == 0 or int(stimno) == 1:  # first two stims are altitude
                        stim = 'alt'+stimno
                        if 'high' in f:
                            stim += 'highSF'
                            mov_dict[stim] = temp
                        elif 'low' in f:
                            stim += 'lowSF'
                            mov_dict[stim] = temp
                    elif int(stimno) == 2 or int(stimno) == 3:
                        stim = 'azi'+stimno
                        if 'high' in f:
                            stim += 'highSF'
                            mov_dict[stim] = temp
                        elif 'low' in f:
                            stim += 'lowSF'
                            mov_dict[stim] = temp
        if surface is None:
            print('Surface image not found, extracting from avg movie')
            surface = temp[0,:,:]
        self.surface = surface
        self.mov_dict = mov_dict
        self.dict_keys = ['low', 'high']

    def combine_stims(self, method='max'):
        '''
        read movies frame by frame and combine them by the given method(avg or max pixel value)
        append it to the already existing mov_dict
        '''
        if method in self.mov_dict.keys():
            print('Combination with this method already present in mov_dict, skipping combination with {0}'.format(method))
            return

        combo = {}
        keys = [k for k in self.mov_dict.keys()]

        for k in keys:
            stimno = list(filter(str.isdigit, k))[0]
            if 'alt' in k:
                key = 'alt' + stimno + method
                if 'alt'+stimno not in combo.keys():
                    combo[key] = self.mov_dict[k]
                else:
                    combo[key] = self.combine(combo[key], self.mov_dict[k], method)
            elif 'azi' in k:
                key = 'azi' + stimno + method
                if 'azi'+stimno not in combo.keys():
                    combo[key] = self.mov_dict[k]
                else:
                    combo[key] = self.combine(combo[key], self.mov_dict[k], method)

        self.dict_keys.append(method)
        self.mov_dict.update(combo)

    def clean_patches(self, patch):
        patch_area = np.count_nonzero(patch)
        if patch_area <= self.params['areaThresh']:
            patch = np.zeros_like(patch)
        return patch

    def get_phasemap(self, plotflag=0):
        phases = {}
        fft_movs = {k: fft_get_phase(fft_movie(v, output_raw=True)) for k, v in self.mov_dict.items()}
        key_list = [k for k in fft_movs.keys()]
        temp = [v for v in fft_movs.values()]
        for j in range(0, len(temp), 2):
            # combine opposite directions[0,1 and 2,3]
            # print('combining {0} and {1}'.format(key_list[j],key_list[j+1]))
            phases[key_list[j]] = ni.gaussian_filter(temp[j]/temp[j+1], 0.8)

        if plotflag:
            self.show_maps(phases, 'phase map')
        return phases

    def get_signmap(self, plotflag=0):
        signmap = {}
        for t in self.dict_keys:
            keys = [k for k in self.phases.keys() if t in k]

            phasemap1 = self.phases[keys[0]]
            phasemap2 = self.phases[keys[1]]
            gradmap1 = np.gradient(phasemap1)
            gradmap2 = np.gradient(phasemap2)
            import scipy.ndimage as ni
            graddir1 = np.zeros(np.shape(gradmap1[0]))
            graddir2 = np.zeros(np.shape(gradmap2[0]))
            for i in range(phasemap1.shape[0]):
                for j in range(phasemap2.shape[1]):
                    graddir1[i, j] = math.atan2(gradmap1[1][i, j], gradmap1[0][i, j])
                    graddir2[i, j] = math.atan2(gradmap2[1][i, j], gradmap2[0][i, j])
            vdiff = np.multiply(np.exp(1j * graddir1), np.exp(-1j * graddir2))
            areamap = np.sin(np.angle(vdiff))
            signmap[t] = ni.gaussian_filter(areamap, 8)
        if plotflag:
            self.show_maps(signmap, title='sign map', nrow=1)
        return signmap

    def get_patchmap(self, plotflag=1):
        signthresh = self.params['signThresh']
        openIter = self.params['openIter']
        patchmap ={}
        patch_CoM = {}
        for key, smap in self.signmaps.items():
            thresh = signthresh
            patchnumold = patchnum = 0
            print('Patching signmap {0}'.format(key))
            # while 4 >= patchnum or patchnum >= 8:
                # increasing the thresh makes the patches bigger, decreasing patch count
                # decreasing the thresh makes the patches smaller, increasing the patch count
            pmap, labeledpatches, patchnum = self.find_patches(smap, thresh, openIter)
            print('{0} patches found with threshold {1}'.format(patchnum, thresh))
            pdiff = patchnum-patchnumold
                # if pdiff > 0:
                #     # patch count increased, increase thresh
                #     thresh += 0.03
                # else:
                #     # patch count decreased or stable, decrease thresh
                #     thresh -= 0.05

            finalpatches = np.zeros(labeledpatches.shape, dtype=np.int)
            # loops for each patch
            for i in range(1, patchnum):
                temp = []
                currpatch = np.zeros(labeledpatches.shape, dtype=np.int)
                currpatch[labeledpatches == i] = 1
                currpatch[labeledpatches != i] = 0
                currpatch = self.clean_patches(currpatch)
                if np.sum(np.multiply(currpatch, pmap)[:]) > 0:
                    finalpatches[currpatch == 1] = 1

            patchmap[key] = finalpatches
        if plotflag:
            self.show_maps(patchmap, title='patch map')

        return patchmap

    def show_maps(self, mapdict, title='phase map', nrow=2):
        import matplotlib.pyplot as plt
        # filter
        phase_list = [adaptive_filter(v) for v in mapdict.values()]
        titles = [k+' '+title for k in mapdict.keys()]
        ncol = int(len(phase_list)/nrow)
        fig = plt.figure(figsize=(10, 10))
        for i, phase in enumerate(phase_list):
            ax = fig.add_subplot(nrow, ncol, i + 1)
            ax.imshow(phase, cmap='seismic')
            ax.set_axis_off()
            ax.set_title(titles[i])
        plt.show()

    def overlay_areas(self):
        pass

    def colorize_map(self, map):
        sz = list(map.shape)
        sz.append(3)
        channels = np.zeros(sz)
        _, binedges = np.histogram(map, bins=3)
        channels[:, :, 0] = np.where(map < binedges[1], map, channels[:, :, 0]) * (-1) * 255
        channels[:, :, 2] = np.where(map > binedges[2], map, channels[:, :, 2]) * 255
        mid = np.where(np.logical_and(binedges[1] < map, map < binedges[2]), map, channels[:, :, 1]) * 255
        channels[:, :, 0] += mid
        channels[:, :, 2] += mid
        return channels

    def save_RetinoMap(self, imflag=0):
        # saves the class as a file object(.retino) and analyzed maps as png images
        fname = self.expdate + '_' + self.animalid
        spath = os.path.join(self.save_path, fname)
        # spath = Users/kaan/src/temp/analysis/190829_KC009
        if os.path.exists(spath):
            print('Overwriting to: {0}'.format(spath))
        else:
            print('{0}  folder not found, creating new...'.format(spath))
            os.makedirs(spath)

        # save the class object as serial
        print('Saving retino class')
        retinopath = os.path.join(spath, fname)
        import dill
        # lines = ['{0} = {1}'.format(k, v) for k, v in self.__dict__.items()]
        with open(retinopath + '.retino', 'wb') as retino_file:
            dill.dump(retino_file)

        if imflag:
            # save the images into image folder inside animal folder
            impath = os.path.join(spath, 'maps')
            if not os.path.exists(impath):
                os.makedirs(impath)
            mapvars = {k: v for k, v in vars(self).items() if 'map' in k}
            for kmap, vmap in mapvars.items():
                current_dir = os.path.join(impath, kmap)
                if not os.path.exists(current_dir):
                    os.makedirs(current_dir)
                print('Saving {0}'.format(kmap))
                for k, m in vmap.items():
                    mapname = k+'.png'
                    pic = self.colorize_map(m)
                    cv2.imwrite(os.path.join(current_dir, mapname), pic)
        print('Retino class saved successfully to {0}'.format(spath))


def load_RetinoMap(load_path):
    import dill
    try:
        print('Loading from {0}'.format(load_path))
        with open(load_path,'rb') as retino:
            temp = dill.load(retino)
            print('Loaded!')
    except Exception as e:
        print(e)
    return temp


path = '/Users/kaan/src/temp/grating/190828_KC009_1P_KC'
a = RetinoMaps()
a.init_RetinoMap(path)
