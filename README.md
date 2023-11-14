# VRP-cs182

:warning: **Attention** :warning:
- if you do not want something to be tracked by git, name the file / folder with **NoTrack** or **TEST** in the name.
    - e.g., do NOT add your wandb API key somewhere in the code
- git collabration on Jupyter notebook can be annoying. We suggest use Jupyter notebooks ONLY for personal debugging purpose.

## wandb
We use [wandb](https://wandb.ai/) for experiment tracking. Please create an account and get your API key. Then run the following command to eport your API key to the environment variable.
```
export WANDB_API_KEY=<your_api_key_here>
```
Please let Yi know your username / email so that he can add you to the `ecal_ml4opt` team.\
When train the model, wandb related args are:
```
--no_wandb # if you do not want to use wandb
--run_name <run_name> # this will also be the run_name on wandb
--who <your_initials> # this will be added to the run name, so that it's easier to filter your own experiments
``` 

Now, try to have a test run:
```
# suppose you are at /AM2019
python run.py --graph_size 20 --run_name 'tsp20_test' --wandb_entity ecal_ml4opt --who <your_initials>
```



## dataset
The entire Amazon VRP dataset (3.3 GB -> 1.6 GB only keep used info) is too large to be included in this repository. We only provide a small subset under the folder `/dataset` for debugging purpose, and keep the full dataset on the server where we have access to. We also provide script to download the full dataset on your server / local machine.
```
coming soon ...
```