import argparse
import os
import time
from pprint import pformat

import jax
import jax.numpy as jnp
import optax

import orbax.checkpoint as ocp
import shutil

jax.devices()  # To prevent conflict with torch gpu

from copy import deepcopy

from jvm.builder import (build_dataloader, build_loss_function, build_model,
                         build_optimizer)
from jvm.utils import (check_cfg, cvt_cfgPathToDict, get_logger,
                       loggin_gpu_info, loggin_system_info)

parser = argparse.ArgumentParser(description="Train classification model.")
parser.add_argument("cfg", type=str, help="Path to configuration file.")
parser.add_argument("--work_dir", type=str, help="Path to working directory.")


def main() -> None:
    args = parser.parse_args()
    cfg_path = args.cfg

    cfg = cvt_cfgPathToDict(cfg_path)
    cfg_ckpt = deepcopy(cfg)
    check_cfg(cfg)


    if args.work_dir is not None:
        cfg['work_dir'] = args.work_dir

    os.makedirs(cfg['work_dir'], exist_ok=True)

    logger = get_logger(cfg['work_dir'])
    logger.info(f"Configuration file: {cfg_path}")
    logger.info(f"Configuration: {os.linesep + pformat(cfg)}")
    logger.info(f"JAX devices: {jax.devices()}")

    device = jax.devices()[0]

    loggin_system_info(logger)
    loggin_gpu_info(logger)

    logger.info(f"JAX version: {jax.__version__}")
    logger.info(f"Optax version: {optax.__version__}")

    # -- Initialize dataloader -- #
    start_iteration = 1

    trainloader = build_dataloader(cfg['train_loader'])
    testloader = build_dataloader(cfg['test_loader'])

    def infinite_trainloader():
        while True:
            yield from trainloader

    # -- Initialize model -- #
    model = build_model(cfg['model'])

    dummy_input = next(iter(trainloader))[0]
    dummy_input = jax.device_put(dummy_input, device)

    key = jax.random.key(seed=cfg['seed'])
    params = model.init(key, dummy_input)

    logger.info(f"Model: {cfg['model']['name']}")
    logger.info(f"Model Architecture: {model}")

    # -- Build loss function -- #
    # TODO: add reduce option to loss function
    criterion = build_loss_function(cfg['loss'])

    def loss_fn(params, logits, target):

        logits = model.apply(params, x)
        acc = jnp.mean(jnp.argmax(logits, axis=-1) == y)
        loss = criterion(logits, target)

        loss = jnp.mean(loss, axis=0)
        return loss, acc

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    # -- Initialize optimizer -- #

    optimizer = build_optimizer(cfg['optimizer'])
    opt_state = optimizer.init(params)


    # TODO: fix checkpoint manager if needed
    # https://orbax.readthedocs.io/en/latest/api_refactor.html
    ckpt_folder_time = time.strftime('%Y-%m-%d_%H-%M-%S')
    ckpt_path = os.path.join(cfg['work_dir'], ckpt_folder_time)
    ckpt_path = os.path.abspath(ckpt_path)
    os.makedirs(ckpt_path, exist_ok=True)
    options = ocp.CheckpointManagerOptions()
    mngr = ocp.CheckpointManager(
            directory=ckpt_path,
            item_names=('state', 'params'),
            options=options,
            )

    # -- Training -- #
    start_time = time.time()
    logger.info(f"Start training at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    for i, (x, y) in zip(range(start_iteration, cfg['iterations'] + 1),
                         infinite_trainloader()):

        x, y = jax.device_put((x, y), device)

        (train_loss, train_acc), grads = grad_fn(params, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        learning_rate = opt_state.hyperparams['learning_rate']

        if i % cfg['log_interval'] == 0:
            logger.info(
                f"Iter: {i}, LR:{learning_rate:.6f}, Loss: {train_loss:.4f}, Accuracy: {train_acc * 100:.2f}%"
            )

        # TODO: clean validation code with function
        if i % cfg['validate_interval'] == 0:
            test_acc = 0
            test_loss = 0
            for x, y in testloader:
                x, y = jax.device_put((x, y), device)
                logits = model.apply(params, x)
                test_acc += jnp.mean(jnp.argmax(logits, axis=-1) == y)
                test_loss += jnp.mean(criterion(logits, y))
            test_acc /= len(testloader)
            test_loss /= len(testloader)
            logger.info(
                f"Validation: Loss: {test_loss:.4f}, Accuracy: {test_acc * 100:.2f}%"
            )

        
        # TODO: fix checkpoint manager for multiple items
        # https://orbax.readthedocs.io/en/latest/api_refactor.html
        # Currently, it saves a single item for each checkpoint.


        if i % cfg['checkpoint_interval'] == 0:
            
            mngr.save(i, args=ocp.args.Composite(
                        state=ocp.args.StandardSave(opt_state),
                        params=ocp.args.StandardSave(params))
                    )
            mngr.wait_until_finished()

            

            
            

    end_time = time.time()
    logger.info(f"End training at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)

    logger.info(
        f"Training time: {int(hours):0>2}:{int(minutes):0>2}:{seconds:05.2f}")


if __name__ == "__main__":
    main()
