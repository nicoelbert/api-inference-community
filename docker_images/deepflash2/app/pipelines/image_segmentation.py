from typing import TYPE_CHECKING, Any, Dict, List
from app.pipelines import Pipeline
if TYPE_CHECKING:
    from PIL import Image

# further Imports
import numpy as np
from deepflash2.all import *
from pathlib import Path


#open tasks
## replace token
## switch to single model to real ensemble
## create popper return
## fill docstring correctly
##adjust model name and second test scenario in test case
## try to delete other sample files


################## stuff that needs to be replaced/moved ###################
API_TOKEN = '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'
# deepflash2 config class


def deepflash2_load_models(self,model_id: str, models_dir="./models"):
    #to be replaced
    repo_id="nicoelbert/deepflash2_testrepo"
    model_filename = model_id + '.pth'

    #download pth file
    from huggingface_hub import hf_hub_download
    temp_dir = hf_hub_download(repo_id=repo_id, filename=model_filename)

    #create new dir
    import os
    os.makedirs(models_dir, exist_ok=True)

    #copy pth file
    import shutil
    shutil.copy(temp_dir, f'{models_dir}/{model_filename}')




################## stuff that needs to be done againg ###################

##########################################################################
class ImageSegementationPipeline(Pipeline):
    def __init__(self, model_id: str, SEED = 0, models_dir = './models'):
        # IMPLEMENT_THIS
        # Preload all the elements you are going to need at inference.
        # For instance your model, processors, tokenizer that might be needed.
        # This function is only called once, so do all the heavy processing I/O here

        self.cfg = cfg = Config(random_state=SEED)
        self.models_dir = self.models_dir

        deepflash2_load_models(model_id, self.models_dir)

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
        import os
        os.makedirs('images', exist_ok=True)
        inputs.save('images/this_img.tif')

        # specify paths
        test_data_path = './'
        ensemble_path = self.models_dir
        prediction_path = Path('./')

        #create model
        model = EnsembleLearner('images',
                                path=test_data_path,
                                config=self.cfg,
                                ensemble_path=ensemble_path)

        # Predict and save semantic segmentation masks
        model.get_ensemble_results(model.files,
                                     use_tta=True,
                                     export_dir=prediction_path / 'masks')

        # Save uncertainty scores
        df_unc = model.df_ens[['file', 'ensemble', 'n_models', 'uncertainty_score']]
        df_unc.to_csv(prediction_path / 'uncertainty_scores.csv', index=False)

        # Show results scores
        model.show_ensemble_results()

        #uncertainty scores
        #return df_unc
        test_return_list = ["test"]
        return test_return_list


