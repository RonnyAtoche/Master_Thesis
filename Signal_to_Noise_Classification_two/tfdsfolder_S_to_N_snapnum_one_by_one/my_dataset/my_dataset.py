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
                names=['nonmerger', 'merger']),
        }),
        supervised_keys=('image', 'label'),
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    path = '/data/s2614855/Signal_to_Noise_Classification_two/labeled_raw_image_resized_aggregated/snapnum_58_66/'
    return {
        'major_than_0': self._generate_examples(path+'major_than_0/'),
        'major_than_5': self._generate_examples(path+'major_than_5/'),
        'major_than_10': self._generate_examples(path+'major_than_10/'),
        'major_than_15': self._generate_examples(path+'major_than_15/'),
        'major_than_20': self._generate_examples(path+'major_than_20/'),
        'major_than_25': self._generate_examples(path+'major_than_25/'),
        'major_than_30': self._generate_examples(path+'major_than_30/'),
        'major_than_35': self._generate_examples(path+'major_than_35/'),
        'major_than_40': self._generate_examples(path+'major_than_40/'),
        'major_than_45': self._generate_examples(path+'major_than_45/'),
    }

  def _generate_examples(self, path):
    """Yields examples."""
    # TODO(dataset_merger): Yields (key, example) tuples from the dataset
    for img_path in glob.glob(path+'*.fits'):
      # Yields (key, example)
      # print(img_path)
      img = fits.open(img_path, ignore_missing_simple=True)[0].data
      if len(img.shape) < 3:
          img = np.expand_dims(img, axis=0)
      img = img.astype('float32')

      yield img_path, {
            'image': img,
            'label': 'nonmerger' if ('nonmerger' in img_path) else 'merger',
      }