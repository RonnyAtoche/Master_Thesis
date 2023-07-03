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
mean_folder_name = 'group_five'

class MyDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for my_dataset dataset."""

  VERSION = tfds.core.Version('2.5.1')
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
        disable_shuffling=True,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    path = '/scratch/s2614855/JWST/output/fits_resized_normalized_grouped/'
    return {
        '{0}'.format(mean_folder_name): self._generate_examples(path+'{0}/'.format(mean_folder_name)),
    }

  def _generate_examples(self, path):
    """Yields examples."""
    # TODO(dataset_merger): Yields (key, example) tuples from the dataset
    for img_path in glob.glob(path+'*.fits'):

      with open('/scratch/s2614855/JWST/fitsresizednormalized_{0}_items_ordered_list.txt'.format(mean_folder_name), 'a') as fp:
        # write each item on a new line
        fp.write("%s\n" % img_path)
        
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