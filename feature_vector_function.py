import numpy as np
import pandas as pd

from evadb.catalog.catalog_type import NdArrayType
from evadb.functions.abstract.abstract_function import AbstractFunction
from evadb.functions.decorators.decorators import forward, setup
from evadb.functions.decorators.io_descriptors.data_types import PandasDataframe
from evadb.functions.gpu_compatible import GPUCompatible

class FeatureVectorFunction(AbstractFunction, GPUCompatible):
    @property
    def name(self) -> str:
        return "FeatureVectorFunction"

    #@setup(cacheable=False, function_type="FeatureExtraction", batchable=False)
    def setup(self):
        self.device = "cpu"
        import tensorflow as tf
        tf.compat.v1.enable_resource_variables()
        import tensorflow_hub as hub
        self.embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        print("Setup of FeatureVectorFunction Done")

    @forward(
        input_signatures=[],
        output_signatures=[],
    )
    def forward(self, frames: pd.DataFrame) -> pd.DataFrame:
        print("Started creating feature vectors")

        column_list = frames["prompt"].tolist()
        embedding = self.embed(column_list)
        ret = pd.DataFrame()

        new_column = []
        for item in embedding:
            item_to_append = item.numpy().astype('float32')
            new_column.append(item_to_append)

        ret["features"] = new_column

        print("Completed creating feature vectors")
        print(ret["features"])

        return ret

    def to_device(self, device: str):
        self.device = device
        return self