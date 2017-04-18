"""Sea Lion Prognostication Engine

https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count
"""

import sys
import os
from collections import namedtuple, defaultdict
import operator
import glob
import csv
from math import sqrt
import os
import datetime

import numpy as np
import pandas as pd

import PIL
from PIL import Image, ImageDraw, ImageFilter

import skimage
import skimage.io
import skimage.measure
from sklearn.cluster import KMeans, SpectralClustering, AffinityPropagation, MeanShift, DBSCAN, estimate_bandwidth

import shapely
import shapely.geometry
from shapely.geometry import Polygon

import matplotlib.pyplot as plt

# Notes
# cls -- sea lion class
# tid -- train, train dotted, or test image id
# _nb -- short for number
# x, y -- don't forget image arrays organized row, col, channels
#
# With contributions from @bitsofbits ...
#


# ================ Meta ====================
__description__ = 'Sea Lion Prognostication Engine'
__version__ = '0.1.0'
__license__ = 'MIT'
__author__ = 'Gavin Crooks (@threeplusone)'
__status__ = "Prototype"
__copyright__ = "Copyright 2017"

# python -c 'import sealiondata; sealiondata.package_versions()'
def package_versions():
    print('sealionengine \t', __version__)
    print('python        \t', sys.version[0:5])
    print('numpy         \t', np.__version__)
    print('skimage       \t', skimage.__version__)
    print('pillow (PIL)  \t', PIL.__version__)
    print('shapely       \t', shapely.__version__)


DATADIR = os.getcwd()

SOURCEDIR = os.path.join(DATADIR, '..', 'input')

VERBOSITY = namedtuple('VERBOSITY', ['QUITE', 'NORMAL', 'VERBOSE', 'DEBUG'])(0,1,2,3)


SeaLionCoord = namedtuple('SeaLionCoord', ['tid', 'cls', 'x', 'y'])

BoundingBox = namedtuple('BoundingBox', ['tid', 'x1', 'y1', 'x2', 'y2'])

YoloBox = namedtuple('YoloBox', ['tid', 'x_ratio', 'y_ratio', 'w_ratio', 'h_ratio'])


class SeaLionData(object):

    def __init__(self, sourcedir=SOURCEDIR, datadir=DATADIR, verbosity=VERBOSITY.NORMAL):
        self.sourcedir = sourcedir
        self.datadir = datadir
        self.verbosity = verbosity

        self.cls_nb = 5

        self.cls_names = (
            'adult_males',
            'subadult_males',
            'adult_females',
            'juveniles',
            'pups',
            'NOT_A_SEA_LION')

        self.cls = namedtuple('ClassIndex', self.cls_names)(*range(0,6))

        # backported from @bitsofbits. Average actual color of dot centers.
        self.cls_colors = (
            (243,8,5),          # red
            (244,8,242),        # magenta
            (87,46,10),         # brown
            (25,56,176),        # blue
            (38,174,21),        # green
            )


        self.dot_radius = 3

        self.train_nb = 947

        self.test_nb = 18636

        self.paths = {
            # Source paths
            'sample'     : os.path.join(sourcedir, 'sample_submission.csv'),
            'counts'     : os.path.join(sourcedir, 'Train', 'train.csv'),
            'train'      : os.path.join(sourcedir, 'Train', '{tid}.jpg'),
            'dotted'     : os.path.join(sourcedir, 'TrainDotted', '{tid}.jpg'),
            'test'       : os.path.join(sourcedir, 'Test', '{tid}.jpg'),
            # Data paths
            'coords'     : os.path.join(datadir, 'coords.csv'),
            }

        # From MismatchedTrainImages.txt
        self.bad_train_ids = (
            3, 7, 9, 21, 30, 34, 71, 81, 89, 97, 151, 184, 215, 234, 242,
            268, 290, 311, 331, 344, 380, 384, 406, 421, 469, 475, 490, 499,
            507, 530, 531, 605, 607, 614, 621, 638, 644, 687, 712, 721, 767,
            779, 781, 794, 800, 811, 839, 840, 869, 882, 901, 903, 905, 909,
            913, 927, 946)

        self._counts = None


    @property
    def trainshort_ids(self):
        # return (0,1,2,4,5,6,8,10)  # Trainshort1
        return range(41,51)         # Trainshort2

    @property
    def train_ids(self):
        """List of all valid train ids"""
        tids = range(0, self.train_nb)
        tids = list(set(tids) - set(self.bad_train_ids) )  # Remove bad ids
        tids.sort()
        return tids

    @property
    def test_ids(self):
        return range(0, self.test_nb)

    def path(self, name, **kwargs):
        """Return path to various source files"""
        path = self.paths[name].format(**kwargs)
        return path

    @property
    def counts(self) :
        """A map from train_id to list of sea lion class counts"""
        if self._counts is None :
            counts = {}
            fn = self.path('counts')
            with open(fn) as f:
                f.readline()
                for line in f:
                    tid_counts = list(map(int, line.split(',')))
                    counts[tid_counts[0]] = tid_counts[1:]
            self._counts = counts
        return self._counts

    @property
    def pd_counts(self):
        """pandas datafram of train.csv"""
        return pd.read_csv(self.path('counts'))

    def rmse(self, tid_counts) :
        true_counts = self.counts

        error = np.zeros(shape=[5] )

        for tid in tid_counts:
            true_counts = self.counts[tid]
            obs_counts = tid_counts[tid]
            diff = np.asarray(true_counts) - np.asarray(obs_counts)
            error += diff*diff
        #print(error)
        error /= len(tid_counts)
        rmse = np.sqrt(error).sum() / 5
        return rmse

    def load_train(self, tids, target_size=(299, 299)):
        """Return train dataset as numpy array

        target_size -- The size to resize each image
        """
        X = np.empty((len(tids), target_size[0], target_size[1], 3))
        for i, tid in enumerate(tids):
            X[i] = load_train_image(tid, target_size=target_size)
        return X

    def load_train_image(self, train_id, border=0, mask=False, target_size=None):
        """Return image as numpy array

        border -- add a black border of this width around image
        mask -- If true mask out masked areas from corresponding dotted image
        """
        img = self._load_image('train', train_id, border, target_size=target_size)
        if mask :
            # The masked areas are not uniformly black, presumable due to
            # jpeg compression artifacts
            dot_img = self._load_image('dotted', train_id, border).astype(np.uint16).sum(axis=-1)
            img = np.copy(img)
            img[dot_img<40] = 0
        return img


    def load_dotted_image(self, train_id, border=0):
        return self._load_image('dotted', train_id, border)


    def load_test_image(self, test_id, border=0):
        return self._load_image('test', test_id, border)


    def _load_image(self, itype, tid, border=0, target_size=None) :
        fn = self.path(itype, tid=tid)
        img = Image.open(fn)
        if target_size is not None:
            img = img.resize((target_size[1], target_size[0]))
        img = np.asarray(img)
        if border :
            height, width, channels = img.shape
            bimg = np.zeros( shape=(height+border*2, width+border*2, channels), dtype=np.uint8)
            bimg[border:-border, border:-border, :] = img
            img = bimg
        return img


    def coords(self, train_id):
        """Extract coordinates of dotted sealions and return list of SeaLionCoord objects)"""

        # Empirical constants
        MIN_DIFFERENCE = 16
        MIN_AREA = 9
        MAX_AREA = 100
        MAX_AVG_DIFF = 50
        MAX_COLOR_DIFF = 32

        src_img = np.asarray(self.load_train_image(train_id, mask=True), dtype = np.float)
        dot_img = np.asarray(self.load_dotted_image(train_id), dtype = np.float)

        img_diff = np.abs(src_img-dot_img)

        # Detect bad data. If train and dotted images are very different then somethings wrong.
        avg_diff = img_diff.sum() / (img_diff.shape[0] * img_diff.shape[1])
        if avg_diff > MAX_AVG_DIFF: raise ValueError("Image %s is too different" % train_id)

        img_diff = np.max(img_diff, axis=-1)

        img_diff[img_diff<MIN_DIFFERENCE] = 0
        img_diff[img_diff>=MIN_DIFFERENCE] = 255

        sealions = []

        for cls, color in enumerate(self.cls_colors):
            # color search backported from @bitsofbits.
            color_array = np.array(color)[None, None, :]
            has_color = np.sqrt(np.sum(np.square(dot_img * (img_diff > 0)[:,:,None] - color_array), axis=-1)) < MAX_COLOR_DIFF
            contours = skimage.measure.find_contours(has_color.astype(float), 0.5)

            if self.verbosity == VERBOSITY.DEBUG :
                print()
                fn = 'diff_{}_{}.png'.format(train_id,cls)
                print('Saving train/dotted difference: {}'.format(fn))
                Image.fromarray((has_color*255).astype(np.uint8)).save(fn)

            for cnt in contours :
                p = Polygon(shell=cnt)
                area = p.area
                if(area > MIN_AREA and area < MAX_AREA) :
                    y, x= p.centroid.coords[0] # DANGER : skimage and cv2 coordinates transposed?
                    x = int(round(x))
                    y = int(round(y))
                    sealions.append( SeaLionCoord(train_id, cls, x, y) )

        if self.verbosity >= VERBOSITY.VERBOSE :
            counts = [0,0,0,0,0]
            for c in sealions :
                counts[c.cls] +=1
            print()
            print('train_id','true_counts','counted_dots', 'difference', sep='\t')
            true_counts = self.counts[train_id]
            print(train_id, true_counts, counts, np.array(true_counts) - np.array(counts) , sep='\t' )

        if self.verbosity == VERBOSITY.DEBUG :
            img = np.copy(sld.load_dotted_image(train_id))
            r = self.dot_radius
            dy,dx,c = img.shape
            for tid, cls, cx, cy in sealions :
                for x in range(cx-r, cx+r+1) : img[cy, x, :] = 255
                for y in range(cy-r, cy+r+1) : img[y, cx, :] = 255
            fn = 'cross_{}.png'.format(train_id)
            print('Saving crossed dots: {}'.format(fn))
            Image.fromarray(img).save(fn)

        return sealions

    def sealion_kmeans(self, sealions, n_clusters=3):
        # Initialize the  numpy array to hold the coordinates
        pts = np.empty((len(sealions), 2))
        prev_tid = None
        i = 0
        for tid, cls, x, y in sealions:
            if prev_tid is not None and prev_tid != tid:
                # We're done loading all the coordinates for the image
                # Run KMeans
                pts = pts[:i]
                y_pred = KMeans(n_clusters).fit_predict(pts)
                plt.scatter(pts[:, 0], pts[:, 1], c=y_pred)
                plt.title('KMeans Image %s' % tid)
                plt.show()

                y_pred = MeanShift().fit_predict(pts)
                plt.scatter(pts[:, 0], pts[:, 1], c=y_pred)
                plt.title('MeanShift Image {} with Bandwith {}'.format(tid, estimate_bandwidth(pts)))
                plt.show()

                # Reset the pts
                pts = np.empty((len(sealions), 2))
                prev_tid = None
                i = 0

            print((tid, cls, x, y))
            pts[i] = np.array([x, y]).astype(np.float64)
            i += 1
            prev_tid = tid


    def regions(self, sealions, border=0):
        img_lions = defaultdict(list)
        for tid, cls, x, y in sealions:
            img_lions[tid].append((x, y))

        bboxes = []
        for tid in img_lions.keys():
            bboxes.append(BoundingBox(tid,
                          min(pt[0] for pt in img_lions[tid]) - border,
                          min(pt[1] for pt in img_lions[tid]) - border,
                          max(pt[0] for pt in img_lions[tid]) + border,
                          max(pt[1] for pt in img_lions[tid]) + border))
        return bboxes

    def yolo_convert(self, bboxes):

        yolo_boxes = []
        for tid, x1, y1, x2, y2 in bboxes:
            img = self.load_train_image(tid, mask=False)
            dh, dw, tmp = img.shape
            xr = (x2 + x1) / (2. * dw)
            yr = (y2 + y1) / (2. * dh)
            wr = (x2 - x1) / float(dw)
            wr = min(wr, (1 - xr) * 2, 2 * xr) # cut the width if too big
            hr = (y2 - y1) / float(dh)
            hr = min(hr, (1 - yr) * 2, 2 * yr) # cut the height if too big
            yolo_boxes.append(YoloBox(tid, xr, yr, wr, hr))
        return yolo_boxes

    def yolo_files(self, yolo_boxes):

        with open('training_list.txt', 'w') as file_list:
            for tid, xr, yr, wr, hr in sorted(yolo_boxes):
                with open('%s.txt' % tid, 'w') as blist:
                    print(' '.join(('0', str(xr), str(yr), str(wr), str(hr))), file=blist)
                print(self.path('train', tid=tid), file=file_list)


    def save_coords(self, train_ids=None):
        if train_ids is None: train_ids = self.train_ids
        fn = self.path('coords')
        self._progress('Saving sealion coordinates to {}'.format(fn))
        with open(fn, 'w') as csvfile:
            writer =csv.writer(csvfile)
            writer.writerow( SeaLionCoord._fields )
            for tid in train_ids :
                self._progress()
                for coord in self.coords(tid):
                    writer.writerow(coord)
        self._progress('done')

    def load_coords(self, fn=None):
        if fn is None: fn = self.path("coords")
        sealions = []
        self._progress("Loading sealion coords from {}".format(fn))
        with open(fn, 'r') as csvfile:
            self._progress()
            coord_reader = csv.reader(csvfile)
            field_row = next(coord_reader)
            for row in coord_reader:
                sealions.append(SeaLionCoord(*[int(v) for v in row]))
        return sealions

    def save_region_boxes(self, regions):
        self._progress('Saving image chunks...')
        self._progress('\n', verbosity=VERBOSITY.VERBOSE)

        for tid, x1, y1, x2, y2 in regions:
            img = self.load_train_image(tid)
            fn = 'region_{tid}_{x1}_{y1}_{x2}_{y2}.png'.format(tid=tid,
                                                               x1=x1,
                                                               y1=y1,
                                                               x2=x2,
                                                               y2=y2)
            Image.fromarray( img[y1:y2, x1:x2, :]).save(fn)
            self._progress()
        self._progress('done')

    def save_sea_lion_chunks(self, coords, chunksize=128):
        self._progress('Saving image chunks...')
        self._progress('\n', verbosity=VERBOSITY.VERBOSE)

        last_tid = -1

        for tid, cls, x, y in coords :
            if tid != last_tid:
                img = self.load_train_image(tid, border=chunksize//2, mask=True)
                last_tid = tid

            fn = 'chunk_{tid}_{cls}_{x}_{y}_{size}.png'.format(size=chunksize, tid=tid, cls=cls, x=x, y=y)
            self._progress(' Saving '+fn, end='\n', verbosity=VERBOSITY.VERBOSE)
            Image.fromarray( img[y:y+chunksize, x:x+chunksize, :]).save(fn)
            self._progress()
        self._progress('done')

    def create_submission(self, preds, fn=None, classes=['adult_males','subadult_males','adult_females','juveniles,pups']):
        """
        Turns predictions from Keras into a csvfile for submission to kaggle

        fn -- the filename to save to, will default to the string
              'sealions_submission_[NOW].csv' where NOW is the current date and
              time.
        classes -- The columns of the csvfile in the specified order. You should
                   change the order if some columns got mixed up.
        """

        sub = pd.DataFrame(data=preds, columns=classes)
        if fn is None:
            now = str(datetime.datetime.now()).replace(' ', '-')
            fn = 'sealions_submission_%s.csv' % now
        sub.to_csv(fn, index_label='test_id')

        if sld.verbosity >= VERBOSITY.VERBOSE:
            print("Finished writing submission for Kaggle to %s" % fn)



    def _progress(self, string=None, end=' ', verbosity=VERBOSITY.NORMAL):
        if self.verbosity < verbosity: return
        if not string :
            print('.', end='')
        elif string == 'done':
            print(' done')
        else:
            print(string, end=end)
        sys.stdout.flush()

# end SeaLionData

if __name__ == "__main__":
    # Count sea lion dots and compare to truth from train.csv
    sld = SeaLionData()
    sld.verbosity = VERBOSITY.VERBOSE
    # for tid in sld.trainshort_ids:
        # coord = sld.coords(tid)
    # sld.save_coords(train_ids = sld.trainshort_ids)
    coords = sld.load_coords()
    # regions = sld.regions(coords, border=32)
    # sld.save_region_boxes(regions=regions)
    # yolo_regions = sld.yolo_convert(regions)
    # sld.yolo_files(yolo_regions)
    sld.sealion_kmeans(coords, 2)
