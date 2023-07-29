import json
import os
from ast import literal_eval
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

from manner import utils
from manner.data.components.file_utils import (
    to_tsv,
    load_idx_map_as_dict,
    check_integrity
)
from manner.data.components.download_utils import (
    download_path,
    extract_file,
    maybe_download,
)
from manner.data.components.sentiment_annotator import SentimentAnnotator

tqdm.pandas()

log = utils.get_pylogger(__name__)


class MINDDataFrame(Dataset):
    def __init__(
        self,
        data_dir: str,
        size: str,
        mind_urls: Dict[str, str],
        categ_embeddings_url: str,
        categ_embeddings_dirname: str,
        categ_embeddings_fpath: str,
        categ_embedding_dim: int,
        entity_embedding_dim: int,
        entity_freq_threshold: int,
        entity_confidence_threshold: float,
        id2index_filenames: Dict[str, str],
        sentiment_model: str,
        tokenizer_max_length: int,
        train: bool,
        validation: bool,
        download: bool,
    ) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.size = size
        self.mind_urls = mind_urls
        self.validation = validation

        self.categ_embeddings_url = categ_embeddings_url
        self.categ_embeddings_dirname = categ_embeddings_dirname
        self.categ_embeddings_fpath = categ_embeddings_fpath
        self.categ_embedding_dim = categ_embedding_dim

        self.entity_freq_threshold = entity_freq_threshold
        self.entity_confidence_threshold = entity_confidence_threshold
        self.entity_embedding_dim = entity_embedding_dim
        self.id2index_filenames = id2index_filenames

        self.sentiment_annotator = SentimentAnnotator(
                sentiment_model=sentiment_model, 
                tokenizer_max_length=tokenizer_max_length
                )
        
        if train:
            self.data_split = "train"
        else:
            self.data_split = "dev"

        self.dst_dir = os.path.join(self.data_dir, "MIND" + self.size + "_" + self.data_split)

        if download:
            self._download_and_extract()
            self._download_and_extract_embeddings()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. Use download=True to download it.")

        self.news, self.behaviors = self.load_data()

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any]:
        user_bhv = self.behaviors.iloc[idx]

        history = user_bhv["history"]
        candidates = user_bhv["candidates"]
        labels = user_bhv["labels"]

        history = self.news.loc[history]
        candidates = self.news.loc[candidates]
        labels = np.array(labels)

        return history, candidates, labels

    def __len__(self) -> int:
        return len(self.behaviors)

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Loads the parsed news and user behaviors.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
                Tuple of news and behaviors datasets.
        """
        news = self._load_news()
        news = news[
            [
                "title",
                "abstract",
                "title_entities",
                "abstract_entities",
                "category_label",
                "sentiment_label",
                "sentiment_score",
            ]
        ]
        log.info(f"News data size: {len(news)}")

        behaviors = self._load_behaviors()
        log.info(
            f"Behaviors data size for data split {self.data_split}, validation={self.validation}: {len(behaviors)}"
        )

        return news, behaviors

    def _load_news(self) -> pd.DataFrame:
        """Loads the parsed news. If not already parsed, loads and preprocesses the raw news data.

        Args:
            news (pd.DataFrame): Dataframe of news articles.

        Returns:
            pd.DataFrame: Parsed news data.
        """
        parsed_news_file = os.path.join(self.dst_dir, "parsed_news.tsv")

        if check_integrity(parsed_news_file):
            # news data already parsed
            log.info(f"News data already parsed. Loading from {parsed_news_file}.")
            news = pd.read_table(
                filepath_or_buffer=parsed_news_file,
                converters={
                    attribute: literal_eval
                    for attribute in ["title_entities", "abstract_entities"]
                },
            )
            news["abstract"].fillna("", inplace=True)
        else:
            log.info(f"News data not parsed. Loading and preprocessing raw data.")
            columns_names = [
                "nid",
                "category",
                "subcategory",
                "title",
                "abstract",
                "url",
                "title_entities",
                "abstract_entities",
            ]
            news = pd.read_table(
                filepath_or_buffer=os.path.join(self.dst_dir, "news.tsv"),
                header=None,
                names=columns_names,
                usecols=range(len(columns_names)),
            )

            # replace missing values
            news["abstract"].fillna("", inplace=True)
            news["title_entities"].fillna("[]", inplace=True)
            news["abstract_entities"].fillna("[]", inplace=True)

            # compute sentiments
            log.info("Computing sentiments.")
            news['sentiment_predictions'] = news['title'].progress_apply(lambda title: self.sentiment_annotator(title))
            news['sentiment_class'], news['sentiment_score'] = zip(*news['sentiment_predictions'])
            news.drop(columns=['sentiment_predictions'], inplace=True)
            log.info("Sentiments computation completed.")

            if self.data_split == "train":
                # keep only entities with a confidence over the threshold
                self.entity2freq = {}
                self._count_entity_freq(news["title_entities"])
                self._count_entity_freq(news["abstract_entities"])

                # keep only entities with a frequency over the threshold
                self.entity2index = {}
                for entity, freq in self.entity2freq.items():
                    if freq > self.entity_freq_threshold:
                        self.entity2index[entity] = len(self.entity2index) + 1
                entity2index_df = pd.DataFrame(self.entity2index.items(), columns=["entity", "index"])
                to_tsv(entity2index_df, os.path.join(self.dst_dir, self.id2index_filenames['entity2index']))

                # encode categorical aspects
                # categ2index map
                log.info('Constructing categ2index map.')
                news_category = news['category'].drop_duplicates().reset_index(drop=True)
                categ2index = {v: k+1 for k, v in news_category.to_dict().items()}
                categ2index_fpath = os.path.join(self.dst_dir, self.id2index_filenames['categ2index'])
                log.info(f'Saving categ2index map of size {len(categ2index)} in {categ2index_fpath}')
                to_tsv(df=pd.DataFrame(categ2index.items(), columns=['category', 'index']), 
                       fpath=categ2index_fpath)

                # sentiment2index map
                log.info('Constructing sentiment2index map.')
                news_sentiment = news['sentiment_class'].drop_duplicates().reset_index(drop=True)
                sentiment2index = {v: k+1 for k, v in news_sentiment.to_dict().items()}
                sentiment2index_fpath = os.path.join(self.dst_dir, self.id2index_filenames['sentiment2index'])
                log.info(f'Saving sentiment2index map of size {len(sentiment2index)} in {sentiment2index_fpath}')
                to_tsv(df=pd.DataFrame(sentiment2index.items(), columns=['sentiment', 'index']), 
                       fpath=sentiment2index_fpath)
            
            else:
                log.info('Loading indices maps.')
                # load categ2index map
                categ2index_fpath = os.path.join(self.data_dir, 'MIND' + self.size + '_train', self.id2index_filenames['categ2index'])
                categ2index = load_idx_map_as_dict(categ2index_fpath)
                
                # load sentiment2index map
                sentiment2index_fpath = os.path.join(self.data_dir, 'MIND' + self.size + '_train', self.id2index_filenames['sentiment2index'])
                sentiment2index = load_idx_map_as_dict(sentiment2index_fpath)
                
                # load entity2index mapping
                entity2index_df = pd.read_table(
                    os.path.join(self.data_dir, "MIND" + self.size + "_train", self.id2index_filenames["entity2index"])
                )
                self.entity2index = dict(entity2index_df.values.tolist())

            # generate category embeddings
            self._generate_category_embeddings(
                    word2index=categ2index,
                    embeddings_fpath=self.categ_embeddings_fpath,
                    embedding_dim=self.categ_embedding_dim,
                    transformed_embeddings_filename = 'categ_embedding'
                    )

            # transform entity embeddings
            self._transform_entity_embeddings(entity2index_df)

            # encode categorical aspects
            news['category_label'] = news['category'].progress_apply(lambda x: categ2index.get(x, 0))
            news['sentiment_label'] = news['sentiment_class'].progress_apply(lambda x: sentiment2index.get(x, 0))
            
            # preprocess entities
            news["title_entities"] = news["title_entities"].progress_apply(
                lambda row: self._filter_entities(row)
            )
            news["abstract_entities"] = news["abstract_entities"].progress_apply(
                lambda row: self._filter_entities(row)
            )

            # cache processed data
            to_tsv(news, parsed_news_file)

        news = news.set_index("nid", drop=True)

        return news

    def _load_behaviors(self) -> pd.DataFrame:
        """Loads the parsed user behaviors. If not already parsed, loads and preprocesses the raw
        behavior data.

        Args:
            news (pd.DataFrame): Dataframe of news articles.

        Returns:
            pd.DataFrame: Parsed user behavior data.
        """
        file_prefix = ""
        if self.data_split == "train":
            file_prefix = "train_" if not self.validation else "val_"
        parsed_behaviors_file = os.path.join(self.dst_dir, file_prefix + "parsed_behaviors.tsv")

        if check_integrity(parsed_behaviors_file):
            # behaviors data already parsed
            log.info(f"User behaviors data already parsed. Loading from {parsed_behaviors_file}.")
            behaviors = pd.read_table(
                filepath_or_buffer=parsed_behaviors_file,
                converters={
                    "history": lambda x: x.strip("[]").replace("'", "").split(", "),
                    "candidates": lambda x: x.strip("[]").replace("'", "").split(", "),
                    "labels": lambda x: list(map(int, x.strip("[]").split(", "))),
                },
            )
        else:
            log.info(f"User behaviors data not parsed. Loading and preprocessing raw data.")
            columns_names = ["impid", "uid", "time", "history", "impressions"]
            behaviors = pd.read_table(
                filepath_or_buffer=os.path.join(self.dst_dir, "behaviors.tsv"),
                header=None,
                names=columns_names,
                usecols=range(len(columns_names)),
            )

            # preprocess
            behaviors["time"] = pd.to_datetime(behaviors["time"], format="%m/%d/%Y %I:%M:%S %p")
            behaviors["history"] = behaviors["history"].fillna("").str.split()
            behaviors["impressions"] = behaviors["impressions"].str.split()
            behaviors["candidates"] = behaviors["impressions"].apply(
                lambda x: [impression.split("-")[0] for impression in x]
            )
            behaviors["labels"] = behaviors["impressions"].apply(
                lambda x: [int(impression.split("-")[1]) for impression in x]
            )
            behaviors = behaviors.drop(columns=["impressions"])

            # drop interactions of users without history
            count_interactions = len(behaviors)
            behaviors = behaviors[behaviors["history"].apply(len) > 0]
            dropped_interactions = count_interactions - len(behaviors)
            log.info(f"Removed {dropped_interactions} interactions without user history.")

            behaviors = behaviors.reset_index(drop=True)

            if self.data_split == "train":
                log.info("Splitting behavior data into train and validation sets.")
                if not self.validation:
                    # split behaviors into training dataset
                    behaviors = behaviors.loc[behaviors["time"] < "2019-11-14 00:00:00"]
                    behaviors = behaviors.reset_index(drop=True)

                    # compute uid2index map
                    log.info("Constructing uid2index map.")
                    uid2index = {}
                    for idx in tqdm(behaviors.index.tolist()):
                        uid = behaviors.loc[idx]["uid"]
                        if uid not in uid2index:
                            uid2index[uid] = len(uid2index) + 1

                    fpath = os.path.join(self.dst_dir, "uid2index.tsv")
                    log.info(f"Saving uid2index map of size {len(uid2index)} in {fpath}")
                    to_tsv(
                        df=pd.DataFrame(uid2index.items(), columns=["uid", "index"]), fpath=fpath
                    )
                else:
                    # split behaviors into validation dataset
                    behaviors = behaviors.loc[behaviors["time"] >= "2019-11-14 00:00:00"]
                    behaviors = behaviors.reset_index(drop=True)

                    # load uid2index map
                    log.info("Loading uid2index map.")
                    fpath = os.path.join(
                        self.data_dir, "MIND" + self.size + "_train", self.id2index_filenames["uid2index"]
                    )
                    uid2index = load_idx_map_as_dict(fpath)
            else:
                # load uid2index map
                log.info("Loading uid2index map.")
                fpath = os.path.join(self.data_dir, "MIND" + self.size + "_train", self.id2index_filenames["uid2index"])
                uid2index = load_idx_map_as_dict(fpath)

            # map uid to index
            log.info("Mapping uid to index.")
            behaviors["user"] = behaviors["uid"].apply(lambda x: uid2index.get(x, 0))

            behaviors = behaviors[["user", "history", "candidates", "labels"]]

            # cache processed data
            log.info(
                f"Caching parsed behaviors of size {len(behaviors)} to {parsed_behaviors_file}."
            )
            to_tsv(behaviors, parsed_behaviors_file)

        return behaviors

    def _transform_entity_embeddings(self, entity2index: pd.DataFrame):
        entity_embedding = pd.read_table(
            os.path.join(self.dst_dir, "entity_embedding.vec"), header=None
        )
        entity_embedding["vector"] = entity_embedding.iloc[:, 1:101].values.tolist()
        entity_embedding = entity_embedding[[0, "vector"]].rename(columns={0: "entity"})

        merged_df = pd.merge(entity_embedding, entity2index, on="entity").sort_values("index")
        entity_embedding_transformed = np.random.normal(
            size=(len(entity2index) + 1, self.entity_embedding_dim)
        )
        for row in merged_df.itertuples(index=False):
            entity_embedding_transformed[row.index] = row.vector
        np.save(
            os.path.join(self.dst_dir, "entity_embedding"),
            entity_embedding_transformed,
            allow_pickle=True,
        )

    def _count_entity_freq(self, data: pd.Series) -> None:
        for row in tqdm(data):
            for entity in json.loads(row):
                times = len(entity["OccurrenceOffsets"]) * entity["Confidence"]
                if times > 0:
                    if entity["WikidataId"] not in self.entity2freq:
                        self.entity2freq[entity["WikidataId"]] = times
                    else:
                        self.entity2freq[entity["WikidataId"]] += times

    def _filter_entities(self, data: pd.Series) -> List[int]:
        filtered_entities = []
        for entity in json.loads(data):
            if (
                entity["Confidence"] > self.entity_confidence_threshold
                and entity["WikidataId"] in self.entity2index
            ):
                filtered_entities.append(self.entity2index[entity["WikidataId"]])
        return filtered_entities

    def _generate_category_embeddings(self, word2index: Dict[str, int], embeddings_fpath: str, embedding_dim: int, transformed_embeddings_filename: Union[str, None]) -> None:
        """ Loads pretrained embeddings for the words in word_dict.

        Args:
            word2index (Dict[str, int]): word dictionary
            embeddings_fpath (str): the filepath of the embeddings to be loaded
            embedding_dim (int): dimensionality of embeddings
            transformed_embeddings_filename (str): the name of the transformed embeddings file
        """

        embedding_matrix = np.random.normal(size=(len(word2index) + 1, embedding_dim))
        exist_word = set()

        with open(embeddings_fpath, "r") as f:
            for line in tqdm(f):
                linesplit = line.split(" ")
                word = line[0]
                if len(word) != 0:
                    if word in word2index:
                        embedding_matrix[word2index[word]] = np.asarray(list(map(float, linesplit[1:])))
                        exist_word.add(word)
        
        log.info(f'Rate of word missed in pretrained embedding: {(len(exist_word)/len(word2index))}.')

        fpath = os.path.join(self.dst_dir, transformed_embeddings_filename)
        if not check_integrity(fpath):
            log.info(f'Saving category embeddings in {fpath}')
            np.save(fpath, embedding_matrix, allow_pickle=True)

    def _download_and_extract(self) -> None:
        """ Downloads the MIND dataset in the specified size, if not already downloaded, then extracts it."""

        # download the dataset
        url = self.mind_urls[self.size][self.data_split]
        log.info(f"Downloading MIND {self.size} dataset for {self.data_split} from {url} if not cached.")

        with download_path(self.data_dir) as path:
            path = maybe_download(url=url, filename=url.split("/")[-1], work_directory=path)
            log.info(f"Compressed dataset downloaded.")

            # extract the compressed data files
            if not os.path.isdir(self.dst_dir):
                log.info(f"Extracting dataset from {path} into {self.dst_dir}.")
                extract_file(archive_file=path, dst_dir=self.dst_dir, clean_archive=True)
                log.info(f"Dataset extraction completed.")

    def _download_and_extract_embeddings(self) -> None:
        """ Downloads and extracts Glove embeddings, if not already downloaded."""
        log.info(f"Downloading Glove embeddings from {self.categ_embeddings_url}.")

        glove_dst_dir = os.path.join(
                self.data_dir, self.categ_embeddings_dirname
                )

        # download the embeddings
        with download_path(self.data_dir) as path:
            path = maybe_download(
                    url=self.categ_embeddings_url,
                    filename=self.categ_embeddings_url.split('/')[-1],
                    work_directory=path
                    )
            log.info(f'Compressed Glove embeddings downloaded.')

            # extract the compressed file
            if not check_integrity(self.categ_embeddings_fpath):
                log.info(f'Extracting Glove embeddings from {path} in {glove_dst_dir}.')
                extract_file(
                        archive_file=path,
                        dst_dir=glove_dst_dir,
                        clean_archive=False
                        )
                log.info(f'Embeddings extraction completed.')

    def _check_exists(self) -> bool:
        return os.path.isdir(self.dst_dir) and os.listdir(self.dst_dir)
