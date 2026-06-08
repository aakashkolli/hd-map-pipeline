"""Rigid-body transforms for labeled coordinate frames.

Input contract:
    SE3 objects represent a transform from ``source_frame`` to
    ``target_frame``. Spatial arrays passed to methods are point clouds
    with shape ``(N, 3)`` in meters.

Output contract:
    ``transform_points`` returns a new ``(N, 3)`` array in the target
    frame. Inputs are never modified in place.

Coordinate frames:
    ``lidar`` follows KITTI Velodyne convention (x forward, y left, z up).
    ``vehicle`` uses the rear-axle vehicle frame (x forward, y left, z up).
    ``world`` uses ENU convention (x east, y north, z up).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ROTATION_DIM = 3
HOMOGENEOUS_DIM = 4
ORTHOGONAL_ATOL = 1e-4


@dataclass(frozen=True)
class SE3:
    """Rigid transform from a source frame to a target frame.

    Convention:
        The transform converts points FROM ``source_frame`` TO
        ``target_frame`` and is written as ``T_target_source``.

    Attributes:
        rotation: ``(3, 3)`` float64 rotation matrix.
        translation: ``(3,)`` float64 translation vector in target frame.
        source_frame: Name of the input coordinate frame.
        target_frame: Name of the output coordinate frame.
    """

    rotation: np.ndarray
    translation: np.ndarray
    source_frame: str
    target_frame: str

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=np.float64)
        translation = np.asarray(self.translation, dtype=np.float64)

        if rotation.shape != (ROTATION_DIM, ROTATION_DIM):
            raise ValueError(
                f"Expected rotation shape ({ROTATION_DIM}, {ROTATION_DIM}), "
                f"got {rotation.shape} for {self.target_frame}<-{self.source_frame}."
            )
        if translation.shape != (ROTATION_DIM,):
            raise ValueError(
                f"Expected translation shape ({ROTATION_DIM},), got "
                f"{translation.shape} for {self.target_frame}<-{self.source_frame}."
            )
        if not self.source_frame:
            raise ValueError("source_frame must be a non-empty frame label.")
        if not self.target_frame:
            raise ValueError("target_frame must be a non-empty frame label.")

        orthogonality = rotation.T @ rotation
        if not np.allclose(
            orthogonality,
            np.eye(ROTATION_DIM),
            atol=ORTHOGONAL_ATOL,
        ):
            raise ValueError(
                "Rotation matrix is not orthogonal for "
                f"{self.target_frame}<-{self.source_frame}."
            )
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=ORTHOGONAL_ATOL):
            raise ValueError(
                "Rotation matrix determinant must be +1 for "
                f"{self.target_frame}<-{self.source_frame}."
            )

        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Apply this transform to point coordinates.

        Args:
            points: ``(N, 3)`` float32 or float64 point array.
                FRAME: SOURCE frame named by ``self.source_frame``.

        Returns:
            ``(N, 3)`` float64 point array.
                FRAME: TARGET frame named by ``self.target_frame``.
        """
        points_array = np.asarray(points)
        if points_array.ndim != 2 or points_array.shape[1] != ROTATION_DIM:
            raise ValueError(
                f"Expected point array shape (N, {ROTATION_DIM}) in "
                f"{self.source_frame} frame, got {points_array.shape}."
            )

        return points_array.astype(np.float64) @ self.rotation.T + self.translation

    def inverse(self) -> "SE3":
        """Return the inverse transform from target frame to source frame."""
        rotation_inverse = self.rotation.T
        translation_inverse = -(rotation_inverse @ self.translation)
        return SE3(
            rotation=rotation_inverse,
            translation=translation_inverse,
            source_frame=self.target_frame,
            target_frame=self.source_frame,
        )

    def compose(self, other: "SE3") -> "SE3":
        """Compose this transform with a following transform.

        Args:
            other: Transform whose source frame matches this transform's
                target frame.

        Returns:
            Transform equivalent to applying ``self`` first, then ``other``.
        """
        if self.target_frame != other.source_frame:
            raise ValueError(
                "Cannot compose transforms with mismatched frames: "
                f"{self.target_frame} != {other.source_frame}."
            )

        rotation = other.rotation @ self.rotation
        translation = other.rotation @ self.translation + other.translation
        return SE3(
            rotation=rotation,
            translation=translation,
            source_frame=self.source_frame,
            target_frame=other.target_frame,
        )

    @classmethod
    def identity(cls, frame: str) -> "SE3":
        """Return an identity transform for one coordinate frame."""
        return cls(
            rotation=np.eye(ROTATION_DIM),
            translation=np.zeros(ROTATION_DIM),
            source_frame=frame,
            target_frame=frame,
        )

    @classmethod
    def from_matrix(
        cls,
        matrix: np.ndarray,
        *,
        source_frame: str,
        target_frame: str,
    ) -> "SE3":
        """Parse a homogeneous transform matrix.

        Args:
            matrix: ``(4, 4)`` homogeneous transform.
                FRAME: maps SOURCE frame to TARGET frame.
            source_frame: Name of the input coordinate frame.
            target_frame: Name of the output coordinate frame.

        Returns:
            SE3 mapping ``source_frame`` points into ``target_frame``.
        """
        matrix_array = np.asarray(matrix, dtype=np.float64)
        if matrix_array.shape != (HOMOGENEOUS_DIM, HOMOGENEOUS_DIM):
            raise ValueError(
                f"Expected homogeneous matrix shape ({HOMOGENEOUS_DIM}, "
                f"{HOMOGENEOUS_DIM}), got {matrix_array.shape}."
            )

        return cls(
            rotation=matrix_array[:ROTATION_DIM, :ROTATION_DIM],
            translation=matrix_array[:ROTATION_DIM, ROTATION_DIM],
            source_frame=source_frame,
            target_frame=target_frame,
        )

    def as_matrix(self) -> np.ndarray:
        """Return a homogeneous matrix representation of this transform."""
        matrix = np.eye(HOMOGENEOUS_DIM, dtype=np.float64)
        matrix[:ROTATION_DIM, :ROTATION_DIM] = self.rotation
        matrix[:ROTATION_DIM, ROTATION_DIM] = self.translation
        return matrix
