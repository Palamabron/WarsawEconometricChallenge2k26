"""
Focal Loss implementation for imbalanced classification.

FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

Gradients are computed w.r.t. the raw logit score s, as required by GBM frameworks.
"""

import numpy as np


class FocalLoss:
    """
    Focal Loss for binary classification.

    Args:
        gamma: Focusing parameter. Higher values force the model to focus on
               hard-to-classify examples by down-weighting easy ones.
        alpha: Weight for the positive class (typically inverse class frequency).
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.94):
        self.gamma = gamma
        self.alpha = alpha

    def __call__(self, y_true: np.ndarray, y_pred_prob: np.ndarray) -> float:
        """
        Compute mean focal loss.

        Args:
            y_true: True labels (0 or 1).
            y_pred_prob: Predicted probabilities in (0, 1).

        Returns:
            Mean focal loss value.
        """
        p = np.clip(y_pred_prob, 1e-7, 1 - 1e-7)
        p_t = np.where(y_true == 1, p, 1 - p)
        alpha_t = np.where(y_true == 1, self.alpha, 1 - self.alpha)
        loss = -alpha_t * (1 - p_t) ** self.gamma * np.log(p_t)
        return float(np.mean(loss))

    def gradient(self, y_true: np.ndarray, y_pred_prob: np.ndarray) -> np.ndarray:
        """
        Gradient of focal loss w.r.t. the raw logit score s.

        Derivation:
            p_t = p if y=1 else (1-p)
            ∂FL/∂p_t = -α_t * [-γ(1-p_t)^(γ-1)*log(p_t) + (1-p_t)^γ/p_t]
            ∂p_t/∂s  = sign * p * (1-p),  sign = 2y - 1
            ∂FL/∂s   = ∂FL/∂p_t * ∂p_t/∂s

        Args:
            y_true: True labels.
            y_pred_prob: Predicted probabilities (output of sigmoid).

        Returns:
            Gradient array (same shape as inputs).
        """
        p = np.clip(y_pred_prob, 1e-7, 1 - 1e-7)
        p_t = np.where(y_true == 1, p, 1 - p)
        alpha_t = np.where(y_true == 1, self.alpha, 1 - self.alpha)
        sign = 2.0 * y_true - 1.0  # +1 for positive class, -1 for negative

        # ∂FL/∂p_t
        d_fl_d_pt = -alpha_t * (
            -self.gamma * (1 - p_t) ** (self.gamma - 1) * np.log(p_t)
            + (1 - p_t) ** self.gamma / p_t
        )

        # ∂p_t/∂s = sign * p * (1 - p)
        d_pt_d_s = sign * p * (1 - p)

        return d_fl_d_pt * d_pt_d_s

    def hessian(self, y_true: np.ndarray, y_pred_prob: np.ndarray) -> np.ndarray:
        """
        Approximate diagonal Hessian of focal loss w.r.t. the raw logit score s.

        Uses the approximation:
            h_i ≈ α_t * (1 - p_t)^γ * p * (1 - p)

        This drops higher-order terms but guarantees positive-definite hessians
        required for stable Newton updates in gradient boosting.

        Args:
            y_true: True labels.
            y_pred_prob: Predicted probabilities.

        Returns:
            Hessian array (positive values, same shape as inputs).
        """
        p = np.clip(y_pred_prob, 1e-7, 1 - 1e-7)
        p_t = np.where(y_true == 1, p, 1 - p)
        alpha_t = np.where(y_true == 1, self.alpha, 1 - self.alpha)
        hess = alpha_t * (1 - p_t) ** self.gamma * p * (1 - p)
        # Clamp to small positive value to prevent zero hessians
        return np.maximum(hess, 1e-6)


def focal_loss_catboost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gamma: float = 2.0,
    alpha: float = 0.94,
) -> tuple[np.ndarray, np.ndarray]:
    """
    CatBoost-compatible focal loss objective.

    Args:
        y_true: True labels.
        y_pred: Raw predictions (logits).
        gamma: Focusing parameter.
        alpha: Positive class weight.

    Returns:
        Tuple of (gradient, hessian) w.r.t. raw logit scores.
    """
    p = 1.0 / (1.0 + np.exp(-y_pred))
    focal = FocalLoss(gamma=gamma, alpha=alpha)
    return focal.gradient(y_true, p), focal.hessian(y_true, p)


def focal_loss_xgboost(
    y_pred: np.ndarray,
    dtrain: object,
    gamma: float = 2.0,
    alpha: float = 0.94,
) -> tuple[np.ndarray, np.ndarray]:
    """
    XGBoost-compatible focal loss objective.

    XGBoost passes (predt, dtrain) in that order.

    Args:
        y_pred: Raw predictions (logits).
        dtrain: XGBoost DMatrix containing labels.
        gamma: Focusing parameter.
        alpha: Positive class weight.

    Returns:
        Tuple of (gradient, hessian) w.r.t. raw logit scores.
    """
    y_true = dtrain.get_label()
    p = 1.0 / (1.0 + np.exp(-y_pred))
    focal = FocalLoss(gamma=gamma, alpha=alpha)
    return focal.gradient(y_true, p), focal.hessian(y_true, p)
