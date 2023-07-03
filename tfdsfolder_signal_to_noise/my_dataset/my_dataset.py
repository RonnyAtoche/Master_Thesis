"""my_dataset dataset."""

import tensorflow_datasets as tfds
from astropy.io import fits
import tensorflow as tf
import glob
import numpy as np
import pandas as pd

# TODO(my_dataset): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(my_dataset): BibTeX citation
_CITATION = """
"""


class MyDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for my_dataset dataset."""

  VERSION = tfds.core.Version('2.1.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Dataset metadata (homepage, citation,...)."""
    return tfds.core.DatasetInfo(
        builder=self,
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Tensor(shape=(1, 224, 224),dtype=tf.float32),
            'label': tfds.features.ClassLabel(
                names=['merger', 'nonmerger']),
        }),
        supervised_keys=('image', 'label'),
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    path = '/data/s2614855/Intro_Project_Data/Intro_Project/snapnum_78_91_train_val_test/'
    return {
        'train': self._generate_examples(path+'train/'),
        'val': self._generate_examples(path+'val/'),
        'test': self._generate_examples(path+'test/'),
    }

  def zoom_in(self, numpy_array_image, zoom_factor):

    h, w = numpy_array_image.shape
    h_value = int((h-h*zoom_factor)/2)
    w_value = int((w-w*zoom_factor)/2)

    numpy_array_image_zoomed = numpy_array_image[h_value:-(h_value), w_value:-(w_value)]

    return numpy_array_image_zoomed

  def _generate_examples(self, path):
    """Yields examples."""
    # TODO(dataset_merger): Yields (key, example) tuples from the dataset
    for img_path in glob.glob(path+'*.fits'):
      # Yields (key, example)
      # print(img_path)
      img = fits.open(img_path, ignore_missing_simple=True)[0].data
      img = self.zoom_in(img, 0.7)
      if len(img.shape) < 3:
          img = np.expand_dims(img, axis=0)
      img = img.astype('float32')

      yield img_path, {
            'image': img,
            'label': 'nonmerger' if ('nonmerger' in img_path) else 'merger',
      }