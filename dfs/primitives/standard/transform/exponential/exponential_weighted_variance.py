import polars as pl
import polars.selectors as cs

from dfs.primitives.base import TransformPrimitive, TFrame


class ExponentialWeightedVariance(TransformPrimitive):
    """Computes the exponentially weighted moving variance for a series of numbers

    Description:
        Returns the exponentially weighted moving variance for a series of
        numbers. Exactly one of center of mass (com), span, half-life, and
        alpha must be provided. Missing values can be ignored when calculating
        weights by setting 'ignore_nulls' to True.

    Args:
        com (float): Specify decay in terms of center of mass for com >= 0.
            Default is None.

        span (float): Specify decay in terms of span for span >= 1.
            Default is None.

        half_life (float): Specify decay in terms of half-life for half_life > 0.
            Default is None.

        alpha (float): Specify smoothing factor alpha directly. Alpha should be
            greater than 0 and less than or equal to 1. Default is None.

        ignore_nulls (bool): Ignore missing values when calculating weights.
            Default is False.
    """

    name = "exponential_weighted_variance"
    input_types = [cs.numeric()]
    return_type = pl.FLOAT_DTYPES
    use_full_dataframe = True

    def __init__(self, com: float = None, span: float = None, half_life: float = None,
                 alpha: float = None, adjust: bool = True, bias: bool = False,
                 min_periods: int = 1, ignore_nulls: bool = None):
        if all(x is None for x in [com, span, half_life, alpha]):
            com = 0.5
        self.com = com
        self.span = span
        self.half_life = half_life
        self.alpha = alpha
        self.adjust = adjust
        self.bias = bias
        self.min_periods = min_periods
        self.ignore_nulls = ignore_nulls

    def _apply(self, df: TFrame) -> pl.Expr:
        return pl.col(df.columns[0]).ewm_var(
            com=self.com,
            span=self.span,
            half_life=self.half_life,
            alpha=self.alpha,
            adjust=self.adjust,
            bias=self.bias,
            min_periods=self.min_periods,
            ignore_nulls=self.ignore_nulls,
        )
