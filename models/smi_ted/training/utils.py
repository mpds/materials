# Deep learning
import getpass
import glob

# Standard library
import os

import pandas as pd
import torch
from datasets import Dataset, load_dataset

# Data
from pubchem_encoder import Encoder


class MoleculeModule:
    def __init__(self, max_len, dataset, data_path, output_data_path=None):
        super().__init__()
        self.dataset = dataset
        self.data_path = data_path
        self.output_data_path = output_data_path
        self.text_encoder = Encoder(max_len)
        self.paired_mode = output_data_path is not None and output_data_path != ""
        self.output_dataset = None  # Initialize this field

    def prepare_data(self):
        pass

    def get_vocab(self):
        # using home made tokenizer, should look into existing tokenizer
        return self.text_encoder.char2id

    def get_cache(self):
        return self.cache_files

    def setup(self, stage=None):
        # using huggingface dataloader
        # create cache in tmp directory of locale mabchine under the current users name to prevent locking issues

        pubchem_path = {"train": self.data_path}
        if "canonical" in pubchem_path["train"].lower():
            pubchem_script = "./pubchem_canon_script.py"
        else:
            pubchem_script = "./pubchem_script.py"
        zinc_path = "./data/ZINC"

        global dataset_dict

        if "ZINC" in self.dataset or "zinc" in self.dataset:
            zinc_files = [f for f in glob.glob(os.path.join(zinc_path, "*.smi"))]
            for zfile in zinc_files:
                print(zfile)
            self.dataset = {"train": zinc_files}
            dataset_dict = load_dataset(
                "./zinc_script.py",
                data_files=self.dataset,
                cache_dir=os.path.join("/tmp", getpass.getuser(), "zinc"),
                split="train",
                trust_remote_code=True,
            )

        elif "pubchem" in self.dataset:
            dataset_dict = load_dataset(
                pubchem_script,
                data_files=pubchem_path,
                cache_dir=os.path.join("/tmp", getpass.getuser(), "pubchem"),
                split="train",
                trust_remote_code=True,
            )

            # Load output dataset if in paired mode
            if self.paired_mode:
                output_path = {"train": self.output_data_path}
                if "canonical" in output_path["train"].lower():
                    output_script = "./pubchem_canon_script.py"
                else:
                    output_script = "./pubchem_script.py"

                print(f"Loading output dataset from {self.output_data_path}")
                try:
                    self.output_dataset = load_dataset(
                        output_script,
                        data_files=output_path,
                        cache_dir=os.path.join(
                            "/tmp", getpass.getuser(), "pubchem_output"
                        ),
                        split="train",
                        trust_remote_code=True,
                    )
                    print(
                        f"Successfully loaded output dataset with {len(self.output_dataset)} samples"
                    )

                    # Ensure datasets have the same size for paired mode
                    if len(self.output_dataset) != len(dataset_dict):
                        print(
                            f"Warning: Input dataset ({len(dataset_dict)} samples) and "
                            f"output dataset ({len(self.output_dataset)} samples) have different sizes"
                        )
                        # Use the smaller size
                        min_size = min(len(dataset_dict), len(self.output_dataset))
                        dataset_dict = dataset_dict.select(range(min_size))
                        self.output_dataset = self.output_dataset.select(
                            range(min_size)
                        )
                        print(
                            f"Adjusted both datasets to {min_size} samples for paired training"
                        )

                except Exception as e:
                    print(f"Error loading output dataset: {e}")
                    self.output_dataset = None
                    self.paired_mode = False
                    print("Falling back to non-paired mode")

        # elif "both" in self.dataset or "Both" in self.dataset or "BOTH" in self.dataset:
        #     dataset_dict_pubchem = load_dataset(
        #         pubchem_script,
        #         data_files=pubchem_path,
        #         cache_dir=os.path.join("/tmp", getpass.getuser(), "pubchem"),
        #         split="train",
        #         trust_remote_code=True,
        #     )
        #     zinc_files = [f for f in glob.glob(os.path.join(zinc_path, "*.smi"))]
        #     for zfile in zinc_files:
        #         print(zfile)
        #     self.dataset = {"train": zinc_files}
        #     dataset_dict_zinc = load_dataset(
        #         "./zinc_script.py",
        #         data_files=self.dataset,
        #         cache_dir=os.path.join("/tmp", getpass.getuser(), "zinc"),
        #         split="train",
        #         trust_remote_code=True,
        #     )
        #     dataset_dict = concatenate_datasets(
        #         [dataset_dict_zinc, dataset_dict_pubchem]
        #     )
        self.pubchem = dataset_dict
        # print(dataset_dict.cache_files)
        self.cache_files = []

        for cache in dataset_dict.cache_files:
            tmp = "/".join(cache["filename"].split("/")[:4])
            self.cache_files.append(tmp)

        # # DEBUG: Print sample pairs after loading
        # if self.paired_mode and hasattr(self, "output_dataset"):
        #     print("\n=== Dataset Verification ===")
        #     for i in range(min(3, len(self.pubchem))):  # Print first 3 pairs
        #         print(f"Pair {i}:")
        #         print(f"Input:  {self.pubchem[i]['text']}")
        #         print(f"Output: {self.output_dataset[i]['text']}")
        #     print("=" * 40 + "\n")

    def get_output_dataset(self):
        """Returns the output dataset if in paired mode, otherwise returns None"""
        return self.output_dataset if self.paired_mode else None


def get_optim_groups(module):
    # setup optimizer
    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear,)
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in module.named_modules():
        for pn, p in m.named_parameters():
            fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
            if pn.endswith("bias"):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in module.named_parameters()}

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.0},
        {
            "params": [param_dict[pn] for pn in sorted(list(no_decay))],
            "weight_decay": 0.0,
        },
    ]

    return optim_groups
