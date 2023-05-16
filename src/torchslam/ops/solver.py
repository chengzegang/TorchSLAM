from typing import Tuple
import torch
from torch import Tensor

from torch.nn import Module
import torch.nn.functional as F
from loguru import logger


def _dense_to_pairs(x: Tensor, y: Tensor, mask: Tensor, padding: float = 0) -> Tuple[Tensor, Tensor, Tensor]:
    x_pair = []
    y_pair = []
    for x1, x2, m in zip(x, y, mask):
        # print(x1.shape, x2.shape, m.shape)
        indices = torch.nonzero(m, as_tuple=True)
        i = indices[-2]
        j = indices[-1]
        if torch.numel(i) == 0 or torch.numel(j) == 0:
            x_pair.append(torch.full(x[0].size(), torch.nan))
            y_pair.append(torch.full(y[0].size(), torch.nan))
            continue
        x_pair.append(x1[..., i, :])
        y_pair.append(x2[..., j, :])
    x_pair = torch.nested.as_nested_tensor(x_pair).to_padded_tensor(torch.nan)
    y_pair = torch.nested.as_nested_tensor(y_pair).to_padded_tensor(torch.nan)
    mask = torch.isfinite(x_pair).all(dim=-1) & torch.isfinite(y_pair).all(dim=-1)
    x_pair = x_pair.nan_to_num(padding)
    y_pair = y_pair.nan_to_num(padding)
    return x_pair, y_pair, mask


def convert_points_to_homogeneous(points: Tensor) -> Tensor:
    r"""Function that converts points from Euclidean to homogeneous space.

    Args:
        points: the points to be transformed with shape :math:`(*, N, D)`.

    Returns:
        the points in homogeneous coordinates :math:`(*, N, D+1)`.

    Examples:
        >>> input = tensor([[0., 0.]])
        >>> convert_points_to_homogeneous(input)
        tensor([[0., 0., 1.]])
    """
    if not isinstance(points, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(points)}")
    if len(points.shape) < 2:
        raise ValueError(f"Input must be at least a 2D tensor. Got {points.shape}")

    h_points = F.pad(points, [0, 1], "constant", 1.0)
    return h_points


def sampson_epipolar_distance(Fm: Tensor, pts1: Tensor, pts2: Tensor, mask: Tensor, eps: float = 1e-8) -> Tensor:
    """Return Sampson distance for correspondences given the fundamental matrix.

    Args:
        pts1: correspondences from the left images with shape :math:`(*, N, (2|3))`. If they are not homogeneous,
              converted automatically.
        pts2: correspondences from the right images with shape :math:`(*, N, (2|3))`. If they are not homogeneous,
              converted automatically.
        Fm: Fundamental matrices with shape :math:`(*, 3, 3)`. Called Fm to avoid ambiguity with torch.nn.functional.
        eps: Small constant for safe sqrt.

    Returns:
        the computed Sampson distance with shape :math:`(*, N)`.
    """
    if not isinstance(Fm, Tensor):
        raise TypeError(f"Fm type is not a torch.Tensor. Got {type(Fm)}")

    if (len(Fm.shape) < 3) or not Fm.shape[-2:] == (3, 3):
        raise ValueError(f"Fm must be a (*, 3, 3) tensor. Got {Fm.shape}")

    if pts1.shape[-1] == 2:
        pts1 = convert_points_to_homogeneous(pts1)

    if pts2.shape[-1] == 2:
        pts2 = convert_points_to_homogeneous(pts2)

    # From Hartley and Zisserman, Sampson error (11.9)
    # sam =  (x'^T F x) ** 2 / (  (((Fx)_1**2) + (Fx)_2**2)) +  (((F^Tx')_1**2) + (F^Tx')_2**2)) )

    # line1_in_2 = (F @ pts1.transpose(dim0=-2, dim1=-1)).transpose(dim0=-2, dim1=-1)
    # line2_in_1 = (F.transpose(dim0=-2, dim1=-1) @ pts2.transpose(dim0=-2, dim1=-1)).transpose(dim0=-2, dim1=-1)

    # Instead we can just transpose F once and switch the order of multiplication
    logger.debug(f'pts1.shape: {pts1.shape}, pts2.shape: {pts2.shape}, mask.shape: {mask.shape}')
    logger.debug(f'total valid pairs: {mask.sum()}')
    pts1, pts2, pair_mask = _dense_to_pairs(pts1, pts2, mask)
    F_t: Tensor = Fm.transpose(-1, -2)
    line1_in_2: Tensor = pts1 @ F_t  # (B, N, D) @ (B, D, D) -> (B, N, D)
    line2_in_1: Tensor = pts2 @ Fm  # (B, N, D) @ (B, D, D) -> (B, N, D)

    # numerator = (x'^T F x) ** 2
    numerator: Tensor = (pts2 * line1_in_2).sum(dim=-1).pow(2)

    # denominator = (((Fx)_1**2) + (Fx)_2**2)) +  (((F^Tx')_1**2) + (F^Tx')_2**2))
    denominator: Tensor = line1_in_2[..., :2].norm(2, dim=-1).pow(2) + line2_in_1[..., :2].norm(2, dim=-1).pow(2)
    out: Tensor = numerator / denominator
    out_mat = torch.full_like(mask, torch.nan).type_as(out)
    out_mat[mask > 0] = out[pair_mask > 0]
    return out_mat


def spherical_project(XYZ: Tensor) -> Tensor:
    """Converts 3D cartesian coordinates to spherical coordinates.

    Args:
        XYZ (Tensor): Tensor of 3D cartesian coordinates with shape :math:`(B, N, 3)`.
    Returns:
        Tensor: Tensor of 2D spherical coordinates with shape :math:`(B, N, 2)`.
    """
    XYZ = XYZ.view(XYZ.shape[0], -1, XYZ.shape[-1])

    lat = torch.asin(XYZ[..., 2])
    lon = torch.atan2(XYZ[..., 1], XYZ[..., 0])

    x = lon / torch.pi
    y = lat / torch.pi * 2

    xy = torch.stack((x, y), dim=-1)
    return xy


def spherical_project_inverse(xy: Tensor) -> Tensor:
    """Converts 2D spherical coordinates to 3D cartesian coordinates.

    Args:
        xy (Tensor): Tensor of 2D spherical coordinates with shape :math:`(B, N, 2)`.

    Returns:
        Tensor: Tensor of 3D cartesian coordinates with shape :math:`(B, N, 3)`.
    """
    xy = xy.view(xy.shape[0], -1, xy.shape[-1])

    lon = torch.pi * xy[..., 0]
    lat = torch.pi * xy[..., 1] / 2

    XYZ = torch.stack(
        (
            torch.cos(lat) * torch.cos(lon),
            torch.cos(lat) * torch.sin(lon),
            torch.sin(lat),
        ),
        dim=-1,
    )

    return XYZ


def find_fundamental_equirectangular(p1: Tensor, p2: Tensor, mask: Tensor | None = None) -> Tensor:
    p1 = spherical_project_inverse(p1)
    p2 = spherical_project_inverse(p2)

    x1, y1, z1 = torch.chunk(p1, dim=-1, chunks=3)  # Bx1xN
    x2, y2, z2 = torch.chunk(p2, dim=-1, chunks=3)  # Bx1xN

    X = torch.cat([x1 * x2, x1 * y2, z1 * z2, y1 * x2, y1 * y2, y1 * z2, z1 * x2, z1 * y2, z1 * z2], dim=-1)
    if mask is not None:
        X = X.transpose(-2, -1) @ mask.type_as(X) @ X
    else:
        X = X.transpose(-2, -1) @ X
    _, _, V = torch.linalg.svd(X)
    F: Tensor = V[..., -1].reshape(-1, 3, 3)
    return F


def find_fundamental(p1: Tensor, p2: Tensor, mask: Tensor | None = None) -> Tensor:
    r"""Find the fundamental matrix from a set of point correspondences.

    Args:
        p1: The set of points seen from the first camera frame in the camera plane
        p2: The set of points seen from the second camera frame in the camera plane
        mask: The mask to filter out outliers with shape :math:`(B, N, 1)`.

    Returns:
        The fundamental matrix with shape :math:`(B, 3, 3)`.

    """
    x1, y1 = torch.chunk(p1, dim=-1, chunks=2)  # Bx1xN
    x2, y2 = torch.chunk(p2, dim=-1, chunks=2)  # Bx1xN
    ones = torch.ones_like(x1)
    X = torch.cat([x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, ones], dim=-1)
    if mask is not None:
        X = X.transpose(-2, -1) @ mask.type_as(X) @ X
    else:
        X = X.transpose(-2, -1) @ X
    U, S, V = torch.linalg.svd(X)
    F: Tensor = V[..., -1].reshape(-1, 3, 3)
    return F


def cross_product_matrix(x: Tensor) -> Tensor:
    r"""Return the cross_product_matrix symmetric matrix of a vector.

    Args:
        x: The input vector to construct the matrix in the shape :math:`(*, 3)`.

    Returns:
        The constructed cross_product_matrix symmetric matrix with shape :math:`(*, 3, 3)`.
    """
    if not x.shape[-1] == 3:
        raise AssertionError(x.shape)
    # get vector compononens
    x0 = x[..., 0]
    x1 = x[..., 1]
    x2 = x[..., 2]

    # construct the matrix, reshape to 3x3 and return
    zeros = torch.zeros_like(x0)
    cross_product_matrix_flat = torch.stack([zeros, -x2, x1, x2, zeros, -x0, -x1, x0, zeros], dim=-1)
    shape_ = x.shape[:-1] + (3, 3)
    return cross_product_matrix_flat.view(*shape_)


def projections_from_fundamental(F: Tensor) -> Tuple[Tensor, Tensor]:
    r"""Get the projection matrices from the Fundamental Matrix.

    Args:
       F_mat: the fundamental matrix with the shape :math:`(B, 3, 3)`.

    Returns:
        The projection matrices with shape :math:`(B, 3, 4, 2)`.
    """
    if len(F.shape) != 3:
        raise AssertionError(F.shape)
    if F.shape[-2:] != (3, 3):
        raise AssertionError(F.shape)

    R1 = torch.zeros_like(F)
    R1.diagonal(dim1=-2, dim2=-1).fill_(1)
    t1 = torch.zeros_like(F[..., [0]])

    Ft_mat = F.transpose(-2, -1)

    U, S, V = torch.linalg.svd(Ft_mat)
    e2 = V[..., -1]  # Bx3

    R2 = cross_product_matrix(e2) @ F  # Bx3x3
    t2 = e2[..., :, None]  # Bx3x1

    P1 = torch.cat([R1, t1], dim=-1)  # Bx3x4
    P2 = torch.cat([R2, t2], dim=-1)  # Bx3x4

    return P1, P2


def relative_motion_from_fundamental(F: Tensor) -> Tuple[Tensor, Tensor]:
    P1, P2 = projections_from_fundamental(F)
    return relative_camera_motion(
        P1[..., :3, :3], P1[..., :3, 3].view(-1, 3, 1), P2[..., :3, :3], P2[..., :3, 3].view(-1, 3, 1)
    )


def triangulate(P1: Tensor, P2: Tensor, points1: Tensor, points2: Tensor) -> Tensor:
    r"""Reconstructs a bunch of points by triangulation.

    Triangulates the 3d position of 2d correspondences between several images.
    Reference: Internally it uses DLT method from Hartley/Zisserman 12.2 pag.312

    The input points are assumed to be in homogeneous coordinate system and being inliers
    correspondences. The method does not perform any robust estimation.

    Args:
        P1: The projection matrix for the first camera with shape :math:`(*, 3, 4)`.
        P2: The projection matrix for the second camera with shape :math:`(*, 3, 4)`.
        points1: The set of points seen from the first camera frame in the camera plane
          coordinates with shape :math:`(*, N, 2)`.
        points2: The set of points seen from the second camera frame in the camera plane
          coordinates with shape :math:`(*, N, 2)`.

    Returns:
        The reconstructed 3d points in the world frame with shape :math:`(*, N, 3)`.
    """
    if not (len(P1.shape) >= 2 and P1.shape[-2:] == (3, 4)):
        raise AssertionError(P1.shape)
    if not (len(P2.shape) >= 2 and P2.shape[-2:] == (3, 4)):
        raise AssertionError(P2.shape)
    if len(P1.shape[:-2]) != len(P2.shape[:-2]):
        raise AssertionError(P1.shape, P2.shape)
    if not (len(points1.shape) >= 2 and points1.shape[-1] == 2):
        raise AssertionError(points1.shape)
    if not (len(points2.shape) >= 2 and points2.shape[-1] == 2):
        raise AssertionError(points2.shape)
    if len(points1.shape[:-2]) != len(points2.shape[:-2]):
        raise AssertionError(points1.shape, points2.shape)
    if len(P1.shape[:-2]) != len(points1.shape[:-2]):
        raise AssertionError(P1.shape, points1.shape)

    # allocate and construct the equations matrix with shape (*, 4, 4)
    points_shape = max(points1.shape, points2.shape)  # this allows broadcasting
    X = torch.zeros(points_shape[:-1] + (4, 4)).type_as(points1)

    for i in range(4):
        X[..., 0, i] = points1[..., 0] * P1[..., 2:3, i] - P1[..., 0:1, i]
        X[..., 1, i] = points1[..., 1] * P1[..., 2:3, i] - P1[..., 1:2, i]
        X[..., 2, i] = points2[..., 0] * P2[..., 2:3, i] - P2[..., 0:1, i]
        X[..., 3, i] = points2[..., 1] * P2[..., 2:3, i] - P2[..., 1:2, i]

    # 1. Solve the system Ax=0 with smallest eigenvalue
    # 2. Return homogeneous coordinates

    _, _, V = torch.linalg.svd(X)

    points3d_h = V[..., -1]
    points3d: Tensor = points3d_h[..., :-1] / points3d_h[..., -1:]
    return points3d


def triangulate_from_points(p1: Tensor, p2: Tensor, mask: Tensor) -> Tensor:
    f = find_fundamental(p1, p2, mask)
    p1, p2 = projections_from_fundamental(f)
    p_3d = triangulate(p1, p2, p1, p2)
    return p_3d


def find_projection_from_points(p2d: Tensor, p3d: Tensor) -> Tensor:
    r"""
    Args:
        p2d (Tensor): The set of points seen from the first camera frame in the camera plane
            coordinates with shape :math:`(B, N, 2)`.
        p3d (Tensor): The set of points seen from the second camera frame in the camera plane
            coordinates with shape :math:`(B, N, 3)`.
    Returns:
        The projection matrix with shape :math:`(B, 3, 4)`.

    """
    B, N, _ = p2d.shape
    p3d = F.pad(p3d, [0, 1], value=1)
    X = torch.zeros((B, N, 2, 12), dtype=p2d.dtype, device=p2d.device)
    X[:, :, 0, 0:4] = p3d
    X[:, :, 0, 8:12] = -p2d[..., 0:1] * p3d
    X[:, :, 1, 4:8] = p3d
    X[:, :, 1, 8:12] = -p2d[..., 1:2] * p3d
    X = X.view(B, N * 2, 12)
    _, _, V = torch.linalg.svd(X)
    P: Tensor = V[..., -1].reshape(-1, 4, 3).transpose(-1, -2)
    return P


def relative_camera_motion(
    R1: torch.Tensor, t1: torch.Tensor, R2: torch.Tensor, t2: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Compute the relative camera motion between two cameras.

    Given the motion parameters of two cameras, computes the motion parameters of the second
    one assuming the first one to be at the origin. If :math:`T1` and :math:`T2` are the camera motions,
    the computed relative motion is :math:`T = T_{2}T^{−1}_{1}`.

    Args:
        R1: The first camera rotation matrix with shape :math:`(*, 3, 3)`.
        t1: The first camera translation vector with shape :math:`(*, 3, 1)`.
        R2: The second camera rotation matrix with shape :math:`(*, 3, 3)`.
        t2: The second camera translation vector with shape :math:`(*, 3, 1)`.

    Returns:
        A tuple with the relative rotation matrix and
        translation vector with the shape of :math:`[(*, 3, 3), (*, 3, 1)]`.
    """
    if not (len(R1.shape) >= 2 and R1.shape[-2:] == (3, 3)):
        raise AssertionError(R1.shape)
    if not (len(t1.shape) >= 2 and t1.shape[-2:] == (3, 1)):
        raise AssertionError(t1.shape)
    if not (len(R2.shape) >= 2 and R2.shape[-2:] == (3, 3)):
        raise AssertionError(R2.shape)
    if not (len(t2.shape) >= 2 and t2.shape[-2:] == (3, 1)):
        raise AssertionError(t2.shape)

    # compute first the relative rotation
    R = R2 @ R1.transpose(-2, -1)

    # compute the relative translation vector
    t = t2 - R @ t1

    return (R, t)


def to_homography(p: Tensor) -> Tensor:
    return F.pad(p, [0, 1], value=1)


class Solver(Module):
    ...


class FundamentalMatrix(Module):
    def __init__(self, camera_model: str = 'pinhole', **kwargs):
        super().__init__()
        self.camera_model = camera_model

    def forward(self, p1: Tensor, p2: Tensor, mask: Tensor) -> Tensor:
        r"""Estimates the fundamental matrix from a set of point correspondences.

        Args:
            p1: The set of points seen from the first camera frame in the camera plane
              coordinates with shape :math:`(*, N, 2)`.
            p2: The set of points seen from the second camera frame in the camera plane
              coordinates with shape :math:`(*, N, 2)`.
            mask: The mask of valid points with shape :math:`(*, N)`.

        Returns:
            The estimated fundamental matrix with shape :math:`(*, 3, 3)`.
        """
        if self.camera_model == 'pinhole':
            return find_fundamental(p1, p2, mask)
        elif self.camera_model == 'equirectangular':
            return find_fundamental_equirectangular(p1, p2, mask)
        else:
            raise NotImplementedError(self.camera_model)


class Triangulation(Module):
    def forward(self, p1: Tensor, p2: Tensor, mask: Tensor) -> Tensor:
        r"""Estimates the fundamental matrix from a set of point correspondences.

        Args:
            p1: The set of points seen from the first camera frame in the camera plane
              coordinates with shape :math:`(*, N, 2)`.
            p2: The set of points seen from the second camera frame in the camera plane
              coordinates with shape :math:`(*, N, 2)`.
            mask: The mask of valid points with shape :math:`(*, N)`.

        Returns:
            The estimated fundamental matrix with shape :math:`(*, 3, 3)`.
        """
        return triangulate_from_points(p1, p2, mask)
