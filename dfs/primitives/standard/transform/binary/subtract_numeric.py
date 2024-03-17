import polars as pl
import polars.selectors as cs

from dfs.primitives.base import TransformPrimitive, TFrame


class SubtractNumeric(TransformPrimitive):
    """Performs element-wise subtraction of two lists.

    Description:
        Given a list of values X and a list of values
        Y, determine the difference of each value
        in X from its corresponding value in Y.

    Args:
        commutative (bool): determines if Deep Feature Synthesis should
            generate both x - y and y - x, or just one. If True, there is no
            guarantee which of the two will be generated. Defaults to True.

    Notes:
        commutative is True by default since False would result in 2 perfectly
        correlated series.
    """

    name = "subtract_numeric"
    input_types = [(cs.numeric(), cs.numeric())]
    return_type = pl.NUMERIC_DTYPES

    description_template = "the result of {} minus {}"
    commutative = True

    def __init__(self, commutative=True):
        super().__init__()
        self.commutative = commutative

    def _apply(self, df: TFrame) -> pl.Expr:
        return pl.col(df.columns[0]) - pl.col(df.columns[1])

    def _get_name(self, cols: list[str]):
        return "%s - %s" % (cols[0], cols[1])
