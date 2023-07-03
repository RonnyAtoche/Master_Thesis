"""dataset_merger dataset."""

import tensorflow_datasets as tfds
from astropy.io import fits
import tensorflow as tf
import glob
import numpy as np
import pandas as pd

# TODO(dataset_merger): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(dataset_merger): BibTeX citation
_CITATION = """
"""


class Datasetmerger(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for dataset_merger dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }
  
  df = pd.read_csv('/home/s2614855/Projects/intro_project/catalog_merger_challenge_TNG_train.csv')

  def _info(self) -> tfds.core.DatasetInfo:
    """Dataset metadata (homepage, citation,...)."""
    return tfds.core.DatasetInfo(
        builder=self,
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Tensor(shape=(1, 320, 320),dtype=tf.float32),
            'label': tfds.features.ClassLabel(
                names=['merger', 'non-merger']),
        }),
        supervised_keys=('image', 'label'),
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    path = '/data/s2614855/Intro_Project_Data/Intro_Project/snapnum_78_91/'
    return {
        'train': self._generate_examples(path+'train/'),
        'val': self._generate_examples(path+'val/'),
        'test': self._generate_examples(path+'test/'),
    }

  
  def _validate_label(self, img_path_var):
    """Returns True or False wether the file path name correspond to a merger or not"""

    fit_file_name = img_path_var.split('/',6)[6]
    ObjectId_name = fit_file_name.replace('_',' ').replace('.',' ').split()[1]
    ObjectId_number = int(ObjectId_name)

    df_loc = self.df[self.df['ID']==ObjectId_number]

    if ((df_loc['is_major_merger'].values[0]) == 1):
        return True
    else:
        return False

  def _generate_examples(self, path):
    """Yields examples."""
    # TODO(dataset_merger): Yields (key, example) tuples from the dataset
    for img_path in glob.glob(path+'*.fits'):
      # Yields (key, example)
      
      img = fits.open(img_path, ignore_missing_simple=True)[1].data

      if len(img.shape) < 3:
          img = np.expand_dims(img, axis=0)
      img = img.astype('float32')

      yield img_path, {
            'image': img,
            'label': 'merger' if (self._validate_label(img_path)) else 'non-merger',
      }