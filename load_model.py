#this script **should** load the model from a user input checkpoint and 
#make some nice plots ---- TBD 


#imports - do we really need all these? 
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import (
    LightningModule,
    Trainer,
    seed_everything,
)
from src.viz.ou import traj_plot
from src.utils.sampling import generate_traj
from src.utils import lht_utils

#import basemodel
from src.models.base_model import BaseModel

try:
    from jammy.utils.debug import decorate_exception_hook
except ImportError:
    # pylint: disable=ungrouped-imports
    from src.utils.lht_utils import decorate_exception_hook
log = lht_utils.get_logger(__name__)

def load_model(config):

    # Init lightning datamodule
    log.info(
        f"Instantiating datamodule <{config.datamodule.module._target_}>"  # pylint: disable=protected-access
    )
    #instantiate training initial data
    datamodule: LightningDataModule = hydra.utils.instantiate(
        config.datamodule.module, config.datamodule
    )


    log.info(
        f"Instantiating model <{config.model.module._target_}>"  # pylint: disable=protected-access
    )
    #instantiate the model class object
    model: LightningModule = hydra.utils.instantiate(config.model.module, config.model)


    # Init lightning datamodule
    log.info(
        f"Instantiating plots <{config.datamodule.module._target_}>"  # pylint: disable=protected-access
    )
    datamodule: LightningDataModule = hydra.utils.instantiate(
        config.datamodule.module, config.datamodule
    )

    #define the test checkpoint path 
    #test_checkpoint_path = "../../../../../../../projappl/project_2011332/pathintegralsampling/test_checkpoint.ckpt"

    # load params from checkpoint
    #test1 = generate_traj(model.sde_model, 
    #            model.dt, 
    #            model.t_end, 
    #            1)
    model.eval()
    #test1 = sample_traj(1)
    #print(test1)
    #returned initialised model module
    return model 

@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):

    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src.train import train
    from src.utils import lht_utils

    OmegaConf.resolve(config)

    #uncomment to print the config 
    # Pretty print config using Rich library
    #if config.get("print_config"):
    #    lht_utils.print_config(config, resolve=True)

    # Load model
    return load_model(config)    

#this is the loaded Basemodel structure. 
main()
#model = model_class.load_from_checkpoint()


#print(sample_traj(self, 10))
#generate samples
#traj_plot(10, generate_traj(loaded_model, dt=0.01, t_end=1.0, num_sample=2000)
#                , "t", r"$x_t$", title="test", 
#                fsave="img.png")
