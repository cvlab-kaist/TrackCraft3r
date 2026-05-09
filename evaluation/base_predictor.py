"""Base predictor interface."""

from abc import ABC


class BasePredictor(ABC):
    """Base class for predictors used by `eval_worldtrack`.

    Sparse predictors (the default for TrackCraft3R) override
    `predict(images_pil, query_uv, visibility, intrinsics, ...)` and return
    `(T, M, 3)` tracks at each query point.
    """

    @property
    def is_dense(self):
        return False

    def predict(self, images_pil, query_uv, visibility, intrinsics):
        raise NotImplementedError
