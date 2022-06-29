from typing import TYPE_CHECKING, Any, Dict, List
from app.pipelines import Pipeline
if TYPE_CHECKING:
    from PIL import Image

# further Imports
import numpy as np
import torch
from pathlib import Path
from PIL import Image

from huggingface_hub import hf_hub_download
import os
import shutil
import json

import json
from json import JSONEncoder


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

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
    def __init__(self, repo_id:str, filename:str='ensemble.pt'):
        # IMPLEMENT_THIS
        # Preload all the elements you are going to need at inference.
        # For instance your model, processors, tokenizer that might be needed.
        # This function is only called once, so do all the heavy processing I/O here

        ensemble_path = hf_hub_download(repo_id=repo_id, filename=filename)

        self.ensemble = torch.jit.load(ensemble_path)

        # create Ensemble predicter here
        """
        # specify paths
        test_data_path = './'
        self.model = EnsemblePredictor('images',
                               config=cfg,
                               ensemble_path=ensemble_trained_path)
        """


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
        #arr:Union[np.ndarray, torch.Tensor]

        inp = np.array(inputs)
        if len(inp.shape)==2:
            inp = np.expand_dims(inp, -1)
        inp = torch.tensor(inp).float().to(self.device)
        with torch.inference_mode():
            pred, _, _ = self.ensemble(inp)

        pred = pred.cpu().numpy()

        ###############################
        # Result Serialization
        # encode np array result based on https://pynative.com/python-serialize-numpy-ndarray-into-json/
        numpyData = {"array": pred}
        encoded_pred = json.dumps(numpyData, cls=NumpyArrayEncoder)  # use dump() to write array into file

        #empty return
        return encoded_pred



