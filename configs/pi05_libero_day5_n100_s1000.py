# Custom openpi training configuration
#
# Insert this TrainConfig block into:
# openpi/src/openpi/training/config.py
#
# This file is a configuration snippet rather than a standalone script.

TrainConfig(
        name="pi05_libero_day5_n100_s1000",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero_day18_n100",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=1,
        num_train_steps=1000,
        log_interval=1,
        save_interval=499,
        keep_period=499,
        num_workers=0,
        wandb_enabled=False,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10,
            peak_lr=5e-5,
            decay_steps=1000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
