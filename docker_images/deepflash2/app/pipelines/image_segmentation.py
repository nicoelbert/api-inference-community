from typing import TYPE_CHECKING, Any, Dict, List
from app.pipelines import Pipeline
if TYPE_CHECKING:
    from PIL import Image

# further Imports
import numpy as np
from deepflash2.all import *
from pathlib import Path
from PIL import Image

from huggingface_hub import hf_hub_download
import os
import shutil
import json

print("imports complete")

#open tasks
## replace token
## switch to single model to real ensemble
## create popper return
## fill docstring correctly
##adjust model name and second test scenario in test case
## try to delete other sample files

#config file für modelle
#repo und modelid


################## stuff that needs to be replaced/moved ###################
#API_TOKEN = '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'
# deepflash2 config class


def deepflash2_load_models(repo_id: str, models_dir="./models"):
    import traceback
    import sys

    #repo_id is passed as model_id
    #download ensemble_config with filenames
    ensemble_json = hf_hub_download(repo_id=repo_id, filename="ensemble.json")
    with open(ensemble_json, 'r') as f:
        data = json.load(f)
    models = data['models']

    #create models directory
    os.makedirs(models_dir, exist_ok=True)

    for model_fn in models:
        # download pth file
        temp_dir = hf_hub_download(repo_id=repo_id, filename=model_fn)
        shutil.copy(temp_dir, f'{models_dir}/{model_fn}')


################## stuff that needs to be done againg ###################

##########################################################################
class ImageSegementationPipeline():
    def __init__(self, model_id: str, SEED = 0, models_dir = './models'):
        # IMPLEMENT_THIS
        # Preload all the elements you are going to need at inference.
        # For instance your model, processors, tokenizer that might be needed.
        # This function is only called once, so do all the heavy processing I/O here
        self.repo_id = model_id

        self.cfg = Config(random_state=SEED)
        self.models_dir = models_dir

        deepflash2_load_models(self.repo_id, models_dir = self.models_dir)

    def __call__(self, inputs: Image.Image) -> List[Dict[str, Any]]:
        """
        Args:
            inputs (:obj:`PIL.Image`):
                The raw image representation as PIL.
                No transformation made whatsoever from the input. Make all necessary transformations here.
        Return:
            A :obj:`list`:. The list contains items that are dicts should be liked {"label": "XXX", "score": 0.82}
                It is preferred if the returned list is in decreasing `score` order
        """
        # save image into dir
        os.makedirs('images', exist_ok=True)
        inputs.save('images/this_img.png')

        # specify paths
        test_data_path = './'

        #create model
        self.model = EnsembleLearner('images',
                                path=Path(test_data_path),
                                config=self.cfg,
                                ensemble_path=Path(self.models_dir))

        # specify paths
        test_data_path = './'
        prediction_path = Path('./')

        # Predict and save semantic segmentation masks
        g_smx, g_std, g_eng = self.model.get_ensemble_results(self.model.files,
                                     use_tta=True,
                                     export_dir=prediction_path / 'masks')

        self.model.show_ensemble_results()

        pred = np.argmax(g_smx['this_img.png'][:], axis=-1).astype('uint8')

        return pred



