from typing import Any, List, Dict, Iterator, Callable, Iterable
import os
import tqdm
import torch
from torch.cuda.amp import GradScaler, autocast
from omegaconf import DictConfig

from apex.parallel import Reducer

# from torchfly.training.trainer_loop import get_lr
from torchfly.training import TrainerLoop, FlyModel
# from torchfly.training.trainer_loop import get_log_variable
from torchfly.training.callbacks import Callback, CallbackHandler, Events
from torchfly.common import move_to_device, get_rank

import logging

logger = logging.getLogger(__name__)

# pylint:disable=no-member


class KnowledgeDialogTrainerLoop(TrainerLoop):
    def __init__(
        self,
        config: DictConfig,
        model: FlyModel,
        train_dataloader_fn: Callable,
        valid_dataloader_fn: Callable = None,
        test_dataloader_fn: Callable = None
    ):
        super().__init__(config, model, train_dataloader_fn, valid_dataloader_fn, test_dataloader_fn)

    def validate(self):
        # Start Validation
        self.callback_handler.fire_event(Events.VALIDATE_BEGIN)
        # Validation
        self.model.eval()
        # No gradient is needed for validation
        all_pairs = []
        with torch.no_grad():
            pbar = tqdm.tqdm(self.validation_dataloader)
            pbar.mininterval = 2.0
            for batch in pbar:
                # send to cuda device
                batch = move_to_device(batch, self.device)
                # get pairs
                pairs = self.model.predict(batch)
                all_pairs.extend(pairs)
        #breakpoint()
        # store pairs for checking
        torch.save(all_pairs, self.config.training.generation.results_direction)
        self.callback_handler.fire_event(Events.VALIDATE_END)