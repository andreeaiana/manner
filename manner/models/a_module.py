from typing import List, Tuple

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from lightning import LightningModule
from MulticoreTSNE import MulticoreTSNE as TSNE
from pytorch_metric_learning.distances import DotProductSimilarity
from pytorch_metric_learning.losses import SupConLoss
from torchmetrics import MeanMetric, MinMetric

from manner.data.components.mind_batch import MINDNewsBatch
from manner.models.components.news_encoder import MannerNewsEncoder
from manner.data.components.file_utils import load_idx_map_as_dict


class AModule(LightningModule):
    def __init__(
        self,
        temperature: float,
        labels_path: str,
        plm_model: str,
        frozen_layers: List[int],
        dropout_probability: float,
        use_entities: bool,
        pretrained_entity_embeddings_path: str,
        entity_embedding_dim: int,
        num_attention_heads: int,
        query_vector_dim: int,
        text_embedding_dim: int,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        if self.hparams.use_entities:
            # load pretrained entity embeddings
            pretrained_entity_embeddings = torch.from_numpy(
                np.load(pretrained_entity_embeddings_path)
            ).float()
        else:
            pretrained_entity_embeddings = torch.empty(self.hparams.entity_embedding_dim)

        # model components
        self.news_encoder = MannerNewsEncoder(
            plm_model=self.hparams.plm_model,
            frozen_layers=self.hparams.frozen_layers,
            dropout_probability=self.hparams.dropout_probability,
            use_entities=self.hparams.use_entities,
            entity_embeddings=pretrained_entity_embeddings,
            entity_embedding_dim=self.hparams.entity_embedding_dim,
            num_attention_heads=self.hparams.num_attention_heads,
            query_vector_dim=self.hparams.query_vector_dim,
            text_embedding_dim=self.hparams.text_embedding_dim,
        )
        self.tsne = TSNE(n_jobs=8)

        label2index = load_idx_map_as_dict(labels_path)
        labels = list(label2index.keys())
        self.index2label = {v: k for k, v in label2index.items()}
        self.color_map = dict(
            zip(
                labels,
                sns.color_palette(cc.glasbey_light, n_colors=len(labels)),
            )
        )

        # loss function
        distance_func = DotProductSimilarity(normalize_embeddings=False)
        self.criterion = SupConLoss(temperature=self.hparams.temperature, distance=distance_func)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation loss
        self.val_loss_best = MinMetric()

        # collect outputs of `*_step`
        self.keys = ['embeddings', 'labels']
        self.validation_step_outputs = {key: [] for key in self.keys}
        self.test_step_outputs = {key: [] for key in self.keys}
    
    def forward(self, batch: MINDNewsBatch) -> torch.Tensor:
        # encode news
        embeddings = self.news_encoder(batch["news"])

        return embeddings

    def on_train_start(self) -> None:
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_loss_best.reset()

    def model_step(self, batch: MINDNewsBatch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embeddings = self.forward(batch)
        labels = batch["labels"]

        loss = self.criterion(embeddings, labels)

        return loss, embeddings, labels

    def training_step(self, batch: MINDNewsBatch, batch_idx: int):
        loss, embeddings, labels = self.model_step(batch)

        # update and log loss
        self.train_loss(loss)
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        return loss

    def on_train_epoch_end(self) -> None:
        pass

    def validation_step(self, batch: MINDNewsBatch, batch_idx: int):
        loss, embeddings, labels = self.model_step(batch)

        # update and log loss
        self.val_loss(loss)
        self.log(
            "val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        self.validation_step_outputs['embeddings'].append(embeddings) 
        self.validation_step_outputs['labels'].append(labels) 

    def on_validation_epoch_end(self) -> None:
        loss = self.val_loss.compute()  # get current val loss
        self.val_loss_best(loss)  # update best so far val loss

        # log `val_loss_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log(
            "val/loss_best",
            self.val_loss_best.compute(),
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        if self.current_epoch % 10 == 0:
            embeddings = (
                torch.cat([output for output in self.validation_step_outputs["embeddings"]], dim=0).detach().cpu()
            )
            labels = torch.cat([output for output in self.validation_step_outputs["labels"]], dim=0).cpu().tolist()
            transformed_labels = [self.index2label[label] for label in labels]

            tsne_embeddings = self.tsne.fit_transform(embeddings)

            # plot TSNE embeddings
            fig = plt.figure(figsize=(10, 7))
            ax = sns.scatterplot(
                x=tsne_embeddings[:, 0],
                y=tsne_embeddings[:, 1],
                hue=[label for label in transformed_labels],
                palette=self.color_map,
                legend="full",
            )
            ax.set_title("Val Embeddings tSNE")
            lgd = ax.legend(bbox_to_anchor=(1, 1), loc=2)
            img_path = self.logger.save_dir + "/val_embeddings_" + str(self.current_epoch) + ".jpg"
            plt.savefig(img_path, bbox_extra_artists=(lgd,), bbox_inches="tight", dpi=400)

            self.logger.log_image(key="tSNE Embeddings", images=[img_path])
        
        # clean memory for the next epoch
        for key in self.keys:
            self.validation_step_outputs[key].clear()

    def test_step(self, batch: MINDNewsBatch, batch_idx: int):
        loss, embeddings, labels = self.model_step(batch)

        # update and log loss
        self.test_loss(loss)
        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        self.test_step_outputs['embeddings'].append(embeddings) 
        self.test_step_outputs['labels'].append(labels) 

    def on_test_epoch_end(self) -> None:
        embeddings = (torch.cat([output for output in self.test_step_outputs["embeddings"]], dim=0).detach().cpu())
        labels = torch.cat([output for output in self.test_step_outputs["labels"]], dim=0).cpu().tolist()
        transformed_labels = [self.index2label[label] for label in labels]

        tsne_embeddings = self.tsne.fit_transform(embeddings)

        # plot TSNE embeddings
        fig = plt.figure(figsize=(10, 7))
        ax = sns.scatterplot(
            x=tsne_embeddings[:, 0],
            y=tsne_embeddings[:, 1],
            hue=[label for label in transformed_labels],
            palette=self.color_map,
            legend="full",
        )
        ax.set_title("Test Embeddings tSNE")
        lgd = ax.legend(bbox_to_anchor=(1, 1), loc=2)
        img_path = self.logger.save_dir + "/test_embeddings_" + str(self.current_epoch) + ".jpg"
        plt.savefig(img_path, bbox_extra_artists=(lgd,), bbox_inches="tight", dpi=400)

        self.logger.log_image(key="tSNE Embeddings", images=[img_path])
        
        # clean memory for the next epoch
        for key in self.keys:
            self.test_step_outputs[key].clear()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization."""
        optimizer = self.hparams.optimizer(params=self.parameters())

        return {"optimizer": optimizer}
