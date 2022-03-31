import pytorch_lightning as pl
from omegaconf import OmegaConf

from nemo.collections.asr.models import EncDecCTCModel, EncDecCTCModelBPE
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

@hydra_runner(config_path=".")
def main(cfg):
    if cfg.n_gpus > 0:
        cfg.model.train_ds.batch_size //= cfg.n_gpus

    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg, resolve=True)}')

    pl.utilities.seed.seed_everything(cfg.seed)

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    if "tokenizer" in cfg.model:
        asr_model = EncDecCTCModelBPE(cfg=cfg.model, trainer=trainer)
    else:
        asr_model = EncDecCTCModel(cfg=cfg.model, trainer=trainer)

    # Initialize the weights of the model from another model, if provided via config
    asr_model.maybe_init_from_pretrained_checkpoint(cfg)

    trainer.fit(asr_model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        gpu = 1 if cfg.trainer.gpus != 0 else 0
        test_trainer = pl.Trainer(
            gpus=gpu,
            precision=trainer.precision,
            amp_level=trainer.accelerator_connector.amp_level,
            amp_backend=cfg.trainer.get("amp_backend", "native"),
        )
        if asr_model.prepare_test(test_trainer):
            test_trainer.test(asr_model)

if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter