from typing import Any, List, Dict, Optional

from lightning import LightningDataModule
from pytorch_metric_learning.samplers import MPerClassSampler
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from manner.data.components.adressa_dataframe import AdressaDataFrame
from manner.data.components.adressa_news_dataset import AdressaCollate, AdressaNewsDataset


class AdressaNewsDataModule(LightningDataModule):
    def __init__(
        self,
        seed: int,
        data_dir: str,
        adressa_url: str, 
        adressa_gzip_filename: str, 
        categ_embeddings_url: str,
        categ_embeddings_dirname: str,
        categ_embeddings_fpath: str,
        categ_embedding_dim: int,
        id2index_filenames: Dict[str, str],
        sentiment_model: str,
        tokenizer_name: str,
        tokenizer_use_fast: bool,
        tokenizer_max_length: int,
        train_day: List[int],
        test_day: List[int],
        neg_num: int,
        user_dev_size: float,
        aspect: str,
        samples_per_class: int,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        drop_last: bool,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hparams.tokenizer_name,
            use_fast=self.hparams.tokenizer_use_fast,
            model_max_length=self.hparams.tokenizer_max_length,
        )

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        AdressaDataFrame(
            seed=self.hparams.seed,
            data_dir=self.hparams.data_dir,
            adressa_url=self.hparams.adressa_url,
            adressa_gzip_filename=self.hparams.adressa_gzip_filename,
            categ_embeddings_url=self.hparams.categ_embeddings_url,
            categ_embeddings_dirname=self.hparams.categ_embeddings_dirname,
            categ_embeddings_fpath=self.hparams.categ_embeddings_fpath,
            categ_embedding_dim=self.hparams.categ_embedding_dim,
            id2index_filenames=self.hparams.id2index_filenames,
            sentiment_model=self.hparams.sentiment_model,
            tokenizer_max_length=self.hparams.tokenizer_max_length,
            train_day=self.hparams.train_day,
            test_day=self.hparams.test_day,
            neg_num=self.hparams.neg_num,
            user_dev_size=self.hparams.user_dev_size,
            train=True,
            validation=False,
            download=True,
        )
        AdressaDataFrame(
            seed=self.hparams.seed,
            data_dir=self.hparams.data_dir,
            adressa_url=self.hparams.adressa_url,
            adressa_gzip_filename=self.hparams.adressa_gzip_filename,
            categ_embeddings_url=self.hparams.categ_embeddings_url,
            categ_embeddings_dirname=self.hparams.categ_embeddings_dirname,
            categ_embeddings_fpath=self.hparams.categ_embeddings_fpath,
            categ_embedding_dim=self.hparams.categ_embedding_dim,
            id2index_filenames=self.hparams.id2index_filenames,
            sentiment_model=self.hparams.sentiment_model,
            tokenizer_max_length=self.hparams.tokenizer_max_length,
            train_day=self.hparams.train_day,
            test_day=self.hparams.test_day,
            neg_num=self.hparams.neg_num,
            user_dev_size=self.hparams.user_dev_size,
            train=False,
            validation=False,
            download=True,
        )

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            trainset = AdressaDataFrame(
                seed=self.hparams.seed,
                data_dir=self.hparams.data_dir,
                adressa_url=self.hparams.adressa_url,
                adressa_gzip_filename=self.hparams.adressa_gzip_filename,
                categ_embeddings_url=self.hparams.categ_embeddings_url,
                categ_embeddings_dirname=self.hparams.categ_embeddings_dirname,
                categ_embeddings_fpath=self.hparams.categ_embeddings_fpath,
                categ_embedding_dim=self.hparams.categ_embedding_dim,
                id2index_filenames=self.hparams.id2index_filenames,
                sentiment_model=self.hparams.sentiment_model,
                tokenizer_max_length=self.hparams.tokenizer_max_length,
                train_day=self.hparams.train_day,
                test_day=self.hparams.test_day,
                neg_num=self.hparams.neg_num,
                user_dev_size=self.hparams.user_dev_size,
                train=True,
                validation=False,
                download=False,
            )
            validset = AdressaDataFrame(
                seed=self.hparams.seed,
                data_dir=self.hparams.data_dir,
                adressa_url=self.hparams.adressa_url,
                adressa_gzip_filename=self.hparams.adressa_gzip_filename,
                categ_embeddings_url=self.hparams.categ_embeddings_url,
                categ_embeddings_dirname=self.hparams.categ_embeddings_dirname,
                categ_embeddings_fpath=self.hparams.categ_embeddings_fpath,
                categ_embedding_dim=self.hparams.categ_embedding_dim,
                id2index_filenames=self.hparams.id2index_filenames,
                sentiment_model=self.hparams.sentiment_model,
                tokenizer_max_length=self.hparams.tokenizer_max_length,
                train_day=self.hparams.train_day,
                test_day=self.hparams.test_day,
                neg_num=self.hparams.neg_num,
                user_dev_size=self.hparams.user_dev_size,
                train=True,
                validation=True,
                download=False,
            )
            testset = AdressaDataFrame(
                seed=self.hparams.seed,
                data_dir=self.hparams.data_dir,
                adressa_url=self.hparams.adressa_url,
                adressa_gzip_filename=self.hparams.adressa_gzip_filename,
                categ_embeddings_url=self.hparams.categ_embeddings_url,
                categ_embeddings_dirname=self.hparams.categ_embeddings_dirname,
                categ_embeddings_fpath=self.hparams.categ_embeddings_fpath,
                categ_embedding_dim=self.hparams.categ_embedding_dim,
                id2index_filenames=self.hparams.id2index_filenames,
                sentiment_model=self.hparams.sentiment_model,
                tokenizer_max_length=self.hparams.tokenizer_max_length,
                train_day=self.hparams.train_day,
                test_day=self.hparams.test_day,
                neg_num=self.hparams.neg_num,
                user_dev_size=self.hparams.user_dev_size,
                train=False,
                validation=False,
                download=False,
            )

            self.data_train = AdressaNewsDataset(
                news=trainset.news, 
                behaviors=trainset.behaviors, 
                aspect=self.hparams.aspect
            )
            self.data_val = AdressaNewsDataset(
                news=validset.news, 
                behaviors=validset.behaviors, 
                aspect=self.hparams.aspect
            )
            self.data_test = AdressaNewsDataset(
                news=testset.news, 
                behaviors=testset.behaviors, 
                aspect=self.hparams.aspect
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            collate_fn=AdressaCollate(self.tokenizer),
            sampler=MPerClassSampler(
                labels=self.data_train.labels,
                m=self.hparams.samples_per_class,
                batch_size=self.hparams.batch_size,
                length_before_new_iter=len(self.data_train),
            ),
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=self.hparams.drop_last,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            collate_fn=AdressaCollate(self.tokenizer),
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=self.hparams.drop_last,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            collate_fn=AdressaCollate(self.tokenizer),
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=self.hparams.drop_last,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    _ = AdressaNewsDataModule()
