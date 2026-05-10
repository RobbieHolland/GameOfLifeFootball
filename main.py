import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import hydra
from omegaconf import DictConfig
from ga import run_ga


@hydra.main(config_path="config", config_name="default", version_base=None)
def main(cfg: DictConfig):
    run_ga(cfg)


if __name__ == "__main__":
    main()
