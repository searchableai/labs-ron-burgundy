import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from torchfly.flylogger import FlyLogger
from torchfly.flyconfig import FlyConfig
# from torchfly.training import TrainerLoop
from knowledge_dialog_loop import KnowledgeDialogTrainerLoop
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerFast

from model import KnowledgeDialogFlyModel
import tqdm

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    T5Tokenizer,
    BartTokenizer,
    GPT2Tokenizer,
    HfArgumentParser,
    DataCollator,
    TrainingArguments,
    set_seed,
)


def my_collate(raw_batch):
    if not isinstance(raw_batch, list):
        raw_batch = [raw_batch]
    # print(raw_batch)
    context_tensor = [one["context_tensor"] for one in raw_batch]
    context_mask = [one["context_mask"] for one in raw_batch]
    target_tensor = [one["target_tensor"] for one in raw_batch]
    target_mask = [one["target_mask"] for one in raw_batch]
    
    collate_batch = {"context_tensor": context_tensor,
                     "context_mask": context_mask,
                     "target_tensor": target_tensor,
                     "target_mask": target_mask}
    
    for key in ["context_tensor", "context_mask", "target_mask"]:
        #breakpoint()
        collate_batch[key] = pad_sequence(collate_batch[key], batch_first=True, padding_value=0)
    collate_batch["target_tensor"] = pad_sequence(collate_batch["target_tensor"], batch_first=True, padding_value=-100)
    #breakpoint()
    return collate_batch

class KnowledgeDialogDataset(Dataset):
    """Dataset class for dialog response generation"""
    
    def __init__(self, dataset_type, config):
        
        if dataset_type == "train":
            self.dataset_dir = config.data.dataset_dir.train
            self.dataset_ratio = config.data.dataset_ratio.train
        elif dataset_type == "dev":
            self.dataset_dir = config.data.dataset_dir.dev
            self.dataset_ratio = config.data.dataset_ratio.dev
        elif dataset_type == "test":
            self.dataset_dir = config.data.dataset_dir.test
            self.dataset_ratio = config.data.dataset_ratio.test
        else:
            raise NameError("Please choose datset_type between train, dev, and test.")

        tokenizer = PreTrainedTokenizerFast.from_pretrained(config.model.name)
        
        lens = int(len(torch.load(self.dataset_dir)) * self.dataset_ratio)
        
        self.raw_dataset = torch.load(self.dataset_dir)[:lens]
        
        self.dataset = []
        for one in tqdm.tqdm(self.raw_dataset):
            #breakpoint()
            context_token_info = tokenizer(one["context"], return_tensors='pt')
            context_tensor = context_token_info["input_ids"][0]
            context_mask = context_token_info["attention_mask"][0]
            
            target_token_info = tokenizer(one["target"], return_tensors='pt')
            target_tensor = target_token_info["input_ids"][0]
            target_mask = target_token_info["attention_mask"][0]

            if len(context_tensor) <= config.data.max_token_length:    
                self.dataset.append({"context_tensor": context_tensor,
                                    "context_mask": context_mask,
                                    "target_tensor": target_tensor,
                                    "target_mask": target_mask})
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


def get_data_sampler(config):
    # use config to decide distributed sampler or not
    pass


class DataLoaderHelper:
    def __init__(self, config):
        self.config = config

    def train_loader_fn(self):
        kwargs = {
            'num_workers': 0,
            'pin_memory': True,
        }

        dataset = KnowledgeDialogDataset("train", self.config)

        if "--local_rank" in self.config:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                                                    dataset,
                                                    num_replicas=self.config.num_gpus_per_node,
                                                    rank=self.config["--local_rank"]
                                                    )
        else:
            train_sampler = None

        dataloader = DataLoader(dataset, 
                            batch_size=self.config.training.batch_size,
                            shuffle=False,
                            sampler=train_sampler,
                            collate_fn=my_collate,
                            **kwargs)

        return dataloader

    def valid_loader_fn(self):
        kwargs = {
            'num_workers': 0,
            'pin_memory': True,
        }

        dataset = KnowledgeDialogDataset("dev", self.config)

        dataloader = DataLoader(dataset, 
                            batch_size=self.config.training.batch_size,
                            shuffle=True,
                            collate_fn=my_collate,
                            **kwargs)

        return dataloader

def change_relative_path(config):
    config.data.project_path = os.getcwd()
    config.data.dataset_dir.train = os.path.join(config.data.project_path, config.data.dataset_dir.train)
    config.data.dataset_dir.dev = os.path.join(config.data.project_path, config.data.dataset_dir.dev)
    config.data.dataset_dir.test = os.path.join(config.data.project_path, config.data.dataset_dir.test)
    config.training.generation.results_direction = os.path.join(config.data.project_path, config.training.generation.results_direction)
    if config.flyconfig.run.dir.startswith("./.."):
        path = Path(config.data.project_path).parent
        config.flyconfig.run.dir = os.path.join(path, config.flyconfig.run.dir[5:])

    return config    



def main():
    
    config_path = os.path.join(os.getcwd(), "config/config.yaml")

    config = FlyConfig.load(config_path=config_path)

    config = change_relative_path(config)

    fly_logger = FlyLogger(config)

    data_helper = DataLoaderHelper(config)

    model = KnowledgeDialogFlyModel(config)

    # trainer = TrainerLoop(config, model, train_dataloader_fn=data_helper.train_loader_fn, valid_dataloader_fn=data_helper.valid_loader_fn)
    trainer = KnowledgeDialogTrainerLoop(config, model, train_dataloader_fn=data_helper.train_loader_fn, valid_dataloader_fn=data_helper.valid_loader_fn)

    trainer.train()


if __name__ == "__main__":
    main()
    # launch_distributed(config_path=config_path, worker_fn=main)
