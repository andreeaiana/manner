from typing import List, Tuple

import torch
from lightning import LightningModule
from torch.nn import CrossEntropyLoss
from torch_geometric.utils import to_dense_batch
from torchmetrics import MeanMetric, MetricCollection, MinMetric
from torchmetrics.classification import AUROC
from torchmetrics.retrieval import RetrievalMRR, RetrievalNormalizedDCG

from manner.data.components.mind_batch import MINDRecBatch
from manner.metrics.diversity import Diversity
from manner.metrics.personalization import Personalization
from manner.models.components.click_predictors import DotProduct
from manner.models.components.news_encoder import PLMTextEncoder as NewsEncoder
from manner.models.components.user_encoder import NRMSUserEncoder as UserEncoder


class NRMSPLMModule(LightningModule):
    def __init__(
        self,
        plm_model: str,
        frozen_layers: List[int],
        dropout_probability: float,
        text_embedding_dim: int,
        num_attention_heads: int,
        query_vector_dim: int,
        num_categ_classes: int,
        num_sent_classes: int,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # model components
        self.news_encoder = NewsEncoder(
            plm_model=self.hparams.plm_model,
            frozen_layers=self.hparams.frozen_layers,
            dropout_probability=self.hparams.dropout_probability,
            text_embedding_dim=self.hparams.text_embedding_dim,
            num_attention_heads=self.hparams.num_attention_heads,
            query_vector_dim=self.hparams.query_vector_dim,
        )

        self.user_encoder = UserEncoder(
            news_embedding_dim=self.hparams.text_embedding_dim,
            num_attention_heads=self.hparams.num_attention_heads,
            query_vector_dim=self.hparams.query_vector_dim,
        )

        self.click_predictor = DotProduct()
        self.criterion = CrossEntropyLoss()

        # metric objects for calculating and averaging performance across batches
        metrics = MetricCollection(
            {
                "auc": AUROC(task="binary", num_classes=2),
                "mrr": RetrievalMRR(),
                "ndcg@5": RetrievalNormalizedDCG(k=5),
                "ndcg@10": RetrievalNormalizedDCG(k=10),
            }
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        categ_div_metrics = MetricCollection(
            {
                "categ_div@5": Diversity(num_classes=self.hparams.num_categ_classes, k=5),
                "categ_div@10": Diversity(num_classes=self.hparams.num_categ_classes, k=10),
            }
        )
        sent_div_metrics = MetricCollection(
            {
                "sent_div@5": Diversity(num_classes=self.hparams.num_sent_classes, k=5),
                "sent_div@10": Diversity(num_classes=self.hparams.num_sent_classes, k=10),
            }
        )
        categ_pers_metrics = MetricCollection(
            {
                "categ_pers@5": Personalization(num_classes=self.hparams.num_categ_classes, k=5),
                "categ_pers@10": Personalization(num_classes=self.hparams.num_categ_classes, k=10),
            }
        )
        sent_pers_metrics = MetricCollection(
            {
                "sent_pers@5": Personalization(num_classes=self.hparams.num_sent_classes, k=5),
                "sent_pers@10": Personalization(num_classes=self.hparams.num_sent_classes, k=10),
            }
        )
        self.test_categ_div_metrics = categ_div_metrics.clone(prefix="test/")
        self.test_sent_div_metrics = sent_div_metrics.clone(prefix="test/")
        self.test_categ_pers_metrics = categ_pers_metrics.clone(prefix="test/")
        self.test_sent_pers_metrics = sent_pers_metrics.clone(prefix="test/")

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation loss
        self.val_loss_best = MinMetric()

        # collect outputs of `*_step`
        self.keys = [
                'preds', 'targets', 
                'cand_news_size', 'hist_news_size',
                'target_categories', 'target_sentiments', 
                'hist_categories', 'hist_sentiments'
                ]
        self.training_step_outputs = {key: [] for key in self.keys}
        self.validation_step_outputs = {key: [] for key in self.keys}
        self.test_step_outputs = {key: [] for key in self.keys}

    def forward(self, batch: MINDRecBatch) -> torch.Tensor:
        # encode clicked news
        clicked_news_vector = self.news_encoder(batch["x_hist"]["text"])
        clicked_news_vector_agg, _ = to_dense_batch(clicked_news_vector, batch["batch_hist"])

        # encode candidate news
        candidate_news_vector = self.news_encoder(batch["x_cand"]["text"])
        candidate_news_vector_agg, _ = to_dense_batch(candidate_news_vector, batch["batch_cand"])

        # encode user
        user_vector = self.user_encoder(clicked_news_vector_agg)

        scores = self.click_predictor(
            user_vector.unsqueeze(dim=1), candidate_news_vector_agg.permute(0, 2, 1)
        )

        return scores

    def on_train_start(self) -> None:
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_loss_best.reset()
        self.val_metrics.reset()

    def model_step(self, batch: MINDRecBatch) -> Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
            torch.Tensor]:
        
        scores = self.forward(batch)
        y_true, mask_cand = to_dense_batch(batch["labels"], batch["batch_cand"])
        candidate_categories, _ = to_dense_batch(batch["x_cand"]["category"], batch["batch_cand"])
        candidate_sentiments, _ = to_dense_batch(batch["x_cand"]["sentiment"], batch["batch_cand"])

        clicked_categories, mask_hist = to_dense_batch(
            batch["x_hist"]["category"], batch["batch_hist"]
        )
        clicked_sentiments, _ = to_dense_batch(batch["x_hist"]["sentiment"], batch["batch_hist"])

        loss = self.criterion(scores, y_true)

        # predictions, targets, indexes for metric computation
        preds = torch.cat(
            [scores[n][mask_cand[n]] for n in range(mask_cand.shape[0])], dim=0
        ).detach()
        targets = torch.cat(
            [y_true[n][mask_cand[n]] for n in range(mask_cand.shape[0])], dim=0
        ).long()

        cand_news_size = torch.tensor(
            [torch.where(mask_cand[n])[0].shape[0] for n in range(mask_cand.shape[0])]
        )
        hist_news_size = torch.tensor(
            [torch.where(mask_hist[n])[0].shape[0] for n in range(mask_hist.shape[0])]
        )

        target_categories = torch.cat(
            [candidate_categories[n][mask_cand[n]] for n in range(mask_cand.shape[0])], dim=0
        ).long()
        target_sentiments = torch.cat(
            [candidate_sentiments[n][mask_cand[n]] for n in range(mask_cand.shape[0])], dim=0
        ).long()

        hist_categories = torch.cat(
            [clicked_categories[n][mask_hist[n]] for n in range(mask_hist.shape[0])], dim=0
        ).long()
        hist_sentiments = torch.cat(
            [clicked_sentiments[n][mask_hist[n]] for n in range(mask_hist.shape[0])], dim=0
        ).long()

        return loss, preds, targets, cand_news_size, hist_news_size, target_categories, target_sentiments, hist_categories, hist_sentiments

    def training_step(self, batch: MINDRecBatch, batch_idx: int):
        loss, preds, targets, cand_news_size, _, _, _, _, _ = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        self.training_step_outputs['preds'].append(preds) 
        self.training_step_outputs['targets'].append(targets) 
        self.training_step_outputs['cand_news_size'].append(cand_news_size) 

        return loss

    def on_train_epoch_end(self):
        preds = torch.cat([output for output in self.training_step_outputs["preds"]])
        targets = torch.cat([output for output in self.training_step_outputs["targets"]])
        cand_news_size = torch.cat([output for output in self.training_step_outputs["cand_news_size"]])
        indexes = torch.arange(cand_news_size.shape[0]).repeat_interleave(cand_news_size)

        self.train_metrics(preds, targets, **{"indexes": indexes})
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
       
        # clean memory for the next epoch
        for key in self.keys:
            self.training_step_outputs[key].clear()

    def validation_step(self, batch: MINDRecBatch, batch_idx: int):
        loss, preds, targets, cand_news_size, _, _, _, _, _ = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log(
            "val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        
        self.validation_step_outputs['preds'].append(preds) 
        self.validation_step_outputs['targets'].append(targets) 
        self.validation_step_outputs['cand_news_size'].append(cand_news_size) 

    def on_validation_epoch_end(self):
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
        
        preds = torch.cat([output for output in self.validation_step_outputs["preds"]])
        targets = torch.cat([output for output in self.validation_step_outputs["targets"]])
        cand_news_size = torch.cat([output for output in self.validation_step_outputs["cand_news_size"]])
        indexes = torch.arange(cand_news_size.shape[0]).repeat_interleave(cand_news_size)

        self.val_metrics(preds, targets, **{"indexes": indexes})
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # clean memory for the next epoch
        for key in self.keys:
            self.validation_step_outputs[key].clear()

    def test_step(self, batch: MINDRecBatch, batch_idx: int):
        loss, preds, targets, cand_news_size, hist_news_size, target_categories, target_sentiments, hist_categories, hist_sentiments = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
        )

        self.test_step_outputs['preds'].append(preds) 
        self.test_step_outputs['targets'].append(targets) 
        self.test_step_outputs['cand_news_size'].append(cand_news_size) 
        self.test_step_outputs['hist_news_size'].append(hist_news_size) 
        self.test_step_outputs['target_categories'].append(target_categories) 
        self.test_step_outputs['target_sentiments'].append(target_sentiments) 
        self.test_step_outputs['hist_categories'].append(hist_categories) 
        self.test_step_outputs['hist_sentiments'].append(hist_sentiments) 

    def on_test_epoch_end(self):
        preds = torch.cat([output for output in self.test_step_outputs["preds"]])
        targets = torch.cat([output for output in self.test_step_outputs["targets"]])
        
        target_categories = torch.cat([output for output in self.test_step_outputs["target_categories"]])
        target_sentiments = torch.cat([output for output in self.test_step_outputs["target_sentiments"]])

        hist_categories = torch.cat([output for output in self.test_step_outputs["hist_categories"]])
        hist_sentiments = torch.cat([output for output in self.test_step_outputs["hist_sentiments"]])
        
        cand_news_size = torch.cat([output for output in self.test_step_outputs["cand_news_size"]])
        hist_news_size = torch.cat([output for output in self.test_step_outputs["hist_news_size"]])
        
        cand_indexes = torch.arange(cand_news_size.shape[0]).repeat_interleave(cand_news_size)
        hist_indexes = torch.arange(hist_news_size.shape[0]).repeat_interleave(hist_news_size)
        
        self.test_metrics(preds, targets, **{"indexes": cand_indexes})
        self.test_categ_div_metrics(preds, target_categories, cand_indexes)
        self.test_sent_div_metrics(preds, target_sentiments, cand_indexes)
        self.test_categ_pers_metrics(
            preds, target_categories, hist_categories, cand_indexes, hist_indexes
        )
        self.test_sent_pers_metrics(
            preds, target_sentiments, hist_sentiments, cand_indexes, hist_indexes
        )

        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(
            self.test_categ_div_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log_dict(
            self.test_sent_div_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log_dict(
            self.test_categ_pers_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log_dict(
            self.test_sent_pers_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        
        # clean memory for the next epoch
        for key in self.keys:
            self.test_step_outputs[key].clear()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization."""
        optimizer = self.hparams.optimizer(params=self.parameters())

        return {"optimizer": optimizer}
