from abc import ABCMeta
from dataclasses import dataclass, fields
from pycozo import Client
from typing import List
import uuid
import pandas as pd
import re
import os
from loguru import logger
from torch import Tensor


@dataclass(init=False)
class Database:
    db_dir: str
    dist_thr: float
    feature_dim: int
    topk: int
    ef: int
    radius: float
    _client: Client | None = None

    INITIAL_GRAPH_SCRIPTS = """
    {{
        :create feature {{
            feature_id: Uuid,
            =>
            data: <F32; {feature_dim}>,
        }}
    }}
    {{
        :create keyframe {{
            keyframe_id: Uuid,
            =>
            pose: <F32; 16>,
            xyz: <F32; 3>?,
            keypoint_ids: [Uuid] default [],
        }}
    }}
    {{
        :create keypoint {{
            keypoint_id: Uuid,
            =>
            xy: <F32; 2>,
            keyframe_id: Uuid,
            landmark_id: Uuid,
        }}
    }}
    {{
        :create landmark {{
            landmark_id: Uuid,
            =>
            xyz: <F32; 3>,
            feature_id: Uuid,
            keypoint_ids: [Uuid] default [],
        }}
    }}
    """
    HNSW_GRAPH_INDEX_SCRIPTS = """
    ::hnsw create feature:l2_index {{
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
    GET_NUM_LANDMARKS = """
        ?[count(landmark_id)] := *landmark[landmark_id, xyz, feature_id, keypoint_ids]
    """
    GET_NUM_KEYPOINTS = """
        ?[count(keypoint_id)] := *keypoint[keypoint_id, xy, keyframe_id, landmark_id]
    """
    GET_NUM_KEYFRAMES = """
        ?[count(keyframe_id)] := *keyframe[keyframe_id, pose, keypoint_ids]
    """

    def __init__(self, **kwargs):
        names = set([f.name for f in fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)
        self.__post_init__()

    def __post_init__(self):
        if not os.path.exists(self.db_dir):
            self.create_db_file()

    @property
    def client(self) -> Client:
        if self._client is None:
            self._client = Client('rocksdb', self.db_dir)
        return self._client

    def create_db_file(self):
        initial_graph_scripts = self.INITIAL_GRAPH_SCRIPTS.format(feature_dim=self.feature_dim)
        hnsw_graph_index_scripts = self.HNSW_GRAPH_INDEX_SCRIPTS.format(feature_dim=self.feature_dim)
        self.client.run(initial_graph_scripts)
        self.client.run(hnsw_graph_index_scripts)

    def update(
        self,
        feature_df: pd.DataFrame,
        landmark_df: pd.DataFrame,
        keypoint_df: pd.DataFrame,
        keyframe_df: pd.DataFrame,
    ):
        self.client.put('feature', feature_df)
        logger.debug(f'landmark_df: \n{landmark_df.head(3)}')
        self.client.put('landmark', landmark_df)
        self.client.put('keypoint', keypoint_df)
        self.client.put('keyframe', keyframe_df)

    @property
    def n_landmarks(self) -> int:
        res: pd.DataFrame = self.client.run(self.GET_NUM_LANDMARKS)
        if len(res) == 0:
            return 0
        count: int = res['count(landmark_id)'][0]
        return count

    @property
    def n_keypoints(self) -> int:
        res = self.client.run(self.GET_NUM_KEYPOINTS)
        if len(res) == 0:
            return 0
        count: int = res['count(keypoint_id)'][0]
        return count

    @property
    def n_keyframes(self) -> int:
        res = self.client.run(self.GET_NUM_KEYFRAMES)
        if len(res) == 0:
            return 0
        count: int = res['count(keyframe_id)'][0]
        return count

    def find_closest_landmarks(self, features: List[List[float]]):
        if self.n_landmarks <= 32:
            return [[]]
        approx_closest_landmark = self.APPROX_CLOSEST_TOPK_NN.format(topk=1, ef=self.ef, radius=self.radius)
        res = self.client.run(approx_closest_landmark, params=dict(qs=features))
        xyz = res['xyz']
        return xyz

    def get_landmarks(self) -> pd.DataFrame:
        if self.n_landmarks == 0:
            return pd.DataFrame({'landmark_id': [], 'xyz': [], 'feature_id': [], 'keypoint_ids': []})
        res: pd.DataFrame = self.client.run(
            '?[landmark_id,  xyz, feature_id, keypoint_ids] := *landmark[landmark_id, xyz, feature_id, keypoint_ids]'
        )
        return res

    def get_keyframes(self):
        if self.n_keyframes == 0:
            return pd.DataFrame({'keyframe_id': [], 'pose': [], 'keypoint_ids': []})
        res = self.client.run('?[keyframe_id, pose, keypoint_ids] := *keyframe[keyframe_id, pose, keypoint_ids]')
        return res

    def close(self):
        if hasattr(self, 'client') and self.client is not None:
            self.client.close()

    def __del__(self):
        self.close()


class AutoExceptionMeta(ABCMeta):
    def __new__(mcs, name, bases, namespace, *args, **kwargs):
        argstr = re.findall('[A-Z][^A-Z]*', name)
        argstr = [a.lower() for a in argstr]
        argstr = ' '.join(argstr)
        namespace['__doc__'] = argstr
        return super().__new__(mcs, name, bases, namespace, *args, **kwargs)


class AutoException(Exception, metaclass=AutoExceptionMeta):
    pass


class FailedToConnectDatabase(AutoException):
    pass
