from abc import ABCMeta
from dataclasses import dataclass, fields
from pycozo import Client
from typing import List, Tuple
import uuid
import pandas as pd
import re
import os
from loguru import logger
from torch import Tensor
from ...utils import config


@dataclass(init=False)
class CozoDB:
    _client: Client | None = None

    INITIAL_GRAPH_SCRIPTS = """
    {{
        :create descriptor {{
            descriptor_id: Uuid,
            =>
            data: <F32; {feature_dim}>,
            keypoint_ids: [Uuid] default [],
        }}
    }}
    {{
        :create keyframe {{
            keyframe_id: Uuid,
            =>
            R: <F32; 9>,
            t: <F32; 3>,
            xyz: <F32; 3>,
            keypoint_ids: [Uuid] default [],
        }}
    }}
    {{
        :create keypoint {{
            keypoint_id: Uuid,
            =>
            xy: <F32; 3>?,
            descriptor_id: Uuid,
            keyframe_id: Uuid,
        }}
    }}
    """
    HNSW_GRAPH_INDEX_SCRIPTS = """
    ::hnsw create descriptor:l2_index {{
    dim: {feature_dim},
    m: 50,
    dtype: F32,
    fields: [data],
    distance: L2,
    ef_construction: 20,
    extend_candidates: false,
    keep_pruned_connections: true,
    }}
    """

    APPROX_CLOSEST_TOPK_NN = """
    {{
        r_query[qs] <- [[$qs]]


        ?[dist, landmark_id, feature, xyz] := ~landmark:l2_nn_index{{ landmark_id, feature, xyz |
            query: query,
            k: {topk},
            ef: {ef},
            bind_distance: dist,
            radius: {radius},
        }}, r_query[qs], q in qs, query = vec(q)

    }}
    """

    def __init__(self, **kwargs):
        names = set([f.name for f in fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)
        self.__post_init__()

    def __post_init__(self):
        if not os.path.exists(config.db_dir):
            self.init()

    @property
    def client(self) -> Client:
        if self._client is None:
            self._client = Client('rocksdb', config.db_dir)
        return self._client

    def init(self):
        initial_graph_scripts = self.INITIAL_GRAPH_SCRIPTS.format(feature_dim=config.feature_dim)
        hnsw_graph_index_scripts = self.HNSW_GRAPH_INDEX_SCRIPTS.format(feature_dim=config.feature_dim)
        self.client.run(initial_graph_scripts)
        self.client.run(hnsw_graph_index_scripts)

    def update(
        self,
        descriptor_df: pd.DataFrame | None = None,
        keypoint_df: pd.DataFrame | None = None,
        keyframe_df: pd.DataFrame | None = None,
    ):
        if descriptor_df is None:
            self.client.put('descriptor', descriptor_df)
        if keypoint_df is None:
            self.client.put('keyframe', keypoint_df)
        if keyframe_df is None:
            self.client.put('keypoint', keyframe_df)

    def insert_keyframes(self, keyframes: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]):
        loc, kpts, descs, R, t = keyframes
        kf_id = uuid.uuid4().hex
        desc_ids = [uuid.uuid4().hex for _ in range(len(descs))]
        keypoint_ids = [uuid.uuid4().hex for _ in range(len(kpts))]
        descriptor_df = pd.DataFrame(
            {
                'descriptor_id': desc_ids,
                'data': descs.tolist(),
                'keypoint_ids': keypoint_ids,
            }
        )

        keypoint_df = pd.DataFrame(
            {
                'keypoint_id': keypoint_ids,
                'xy': kpts.tolist(),
                'descriptor_id': desc_ids,
                'keyframe_id': [kf_id] * len(kpts),
            }
        )

        keyframe_df = pd.DataFrame(
            {
                'keyframe_id': [kf_id],
                'xyz': [loc.tolist()],
                'R': [R.tolist()],
                't': [t.tolist()],
                'keypoint_ids': keypoint_ids,
            }
        )
        descriptor_df.to_feather('descriptor.arrow')
        keypoint_df.to_feather('keypoint.arrow')
        keyframe_df.to_feather('keyframe.arrow')

        self.client.put('descriptor', descriptor_df)
        self.client.put('keypoint', keypoint_df)
        self.client.put('keyframe', keyframe_df)

    def close(self):
        if hasattr(self, 'client') and self.client is not None:
            self.client.close()

    def __del__(self):
        self.close()
