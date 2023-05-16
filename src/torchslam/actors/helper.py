from typing import Tuple
import cugraph
import cudf
import pandas as pd
import torch
from torch import Tensor
import uuid
from loguru import logger


def build_dataframes(
    batch_idx: Tensor, descs: Tensor, p2ds: Tensor, p3ds: Tensor, P: Tensor, thr: float = 0.1
) -> Tuple[pd.DataFrame, ...]:
    logger.debug(f'shapes of descs: {descs.shape}, p2ds: {p2ds.shape}, p3ds: {p3ds.shape}, P: {P.shape}')
    B, D = descs.shape

    dists = torch.cdist(p3ds, p3ds)

    graph = dists < thr

    # find connected components, merge these landmarks
    graph = cugraph.from_numpy_array(graph.int().cpu().numpy())
    cc_df: cudf.DataFrame = cugraph.connected_components(graph)

    # build feature dedicated dataframe
    landmark_df = cc_df.groupby('labels', as_index=False).agg({'vertex': 'collect'}).to_pandas()
    landmark_df['feature'] = landmark_df['vertex'].apply(lambda idx: descs[idx].nanmean(0).tolist())
    landmark_df['xyz'] = landmark_df['vertex'].apply(lambda idx: p3ds[idx].nanmean(0).tolist())

    feature_df = landmark_df[['feature']].rename(columns={'feature': 'data'})
    feature_df['feature_id'] = feature_df.index.map(lambda _: uuid.uuid4().hex)
    landmark_df['feature_id'] = feature_df['feature_id']
    landmark_df.drop(columns=['feature'], inplace=True)
    landmark_df['landmark_id'] = landmark_df.index.map(lambda _: uuid.uuid4().hex)

    # build keypoint dataframe
    keypoint_df = pd.DataFrame.from_dict(
        {'keypoint_id': [uuid.uuid4().hex for _ in range(len(p2ds))], 'xy': p2ds.tolist()}
    )
    landmark_df['keypoint_ids'] = landmark_df['vertex'].apply(
        lambda idx: [keypoint_df.iloc[i]['keypoint_id'] for i in idx]
    )

    for _, (lid, kids) in landmark_df[['landmark_id', 'keypoint_ids']].iterrows():
        for kid in kids:
            keypoint_df.loc[keypoint_df['keypoint_id'] == kid, 'landmark_id'] = lid

    # build keyframe dataframe
    keyframe_df = pd.DataFrame.from_dict(
        {'keyframe_id': [uuid.uuid4().hex for _ in range(B)], 'pose': P.view(-1, 16).tolist()}
    )
    keypoint_df['keyframe_index'] = batch_idx.int().flatten().tolist()
    keypoint_df['keyframe_id'] = keypoint_df['keyframe_index'].map(lambda idx: keyframe_df.iloc[idx]['keyframe_id'])  # type: ignore
    keypoint_df.drop(columns=['keyframe_index'], inplace=True)
    keyframe_df['keypoint_ids'] = keyframe_df['keyframe_id'].map(
        lambda kid: keypoint_df[keypoint_df['keyframe_id'] == kid]['keypoint_id'].tolist()
    )
    landmark_df.drop(columns=['vertex', 'labels'], inplace=True)

    feature_df.dropna(inplace=True)
    landmark_df.dropna(inplace=True)
    keypoint_df.dropna(inplace=True)
    keyframe_df.dropna(inplace=True)

    logger.debug(f'Unique landmarks: {len(landmark_df)}')
    logger.debug(
        f'column names: \n feature_df {feature_df.columns} landmark_df: {landmark_df.columns}, keypoint_df: {keypoint_df.columns}, keyframe_df: {keyframe_df.columns}'
    )
    return feature_df, landmark_df, keypoint_df, keyframe_df
