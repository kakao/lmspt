
"""
Launch training scripts
"""
from omegaconf import DictConfig, OmegaConf
from typing import Optional
import os
import hydra
import torch


def train(cfg):
    if hasattr(cfg.trainer, "trainer"):
        trainer = hydra.utils.instantiate(cfg.trainer.trainer)
    else:
        trainer = hydra.utils.instantiate(cfg.trainer)

    OmegaConf.save(config=cfg, f=os.path.join(trainer.exp_dir, "config.yaml"))

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    print(f"RANK: {rank}, WORLD_SIZE: {world_size}")

    train_dataloader = hydra.utils.instantiate(cfg.data.train_loader)
    valid_dataloader = hydra.utils.instantiate(cfg.data.valid_loader)
    trainer._build_dataloader(train_dataloader, valid_dataloader)
    trainer.train_loop()


@hydra.main(
    version_base="1.3",
    config_path="./lmspt/conf",
    config_name="lmspt_train.yaml",
)
def main(cfg: DictConfig) -> Optional[float]:
    # train the model
    train(cfg)


if __name__ == "__main__":
    main(None)
