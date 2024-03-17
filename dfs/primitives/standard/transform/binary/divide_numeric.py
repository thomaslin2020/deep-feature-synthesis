import polars as pl
import polars.selectors as cs

from dfs.primitives.base import TransformPrimitive, TFrame


class DivideNumeric(TransformPrimitive):
    """Performs element-wise division of two lists.

    Description:
        Given a list of values X and a list of values
        Y, determine the quotient of each value in X
        divided by its corresponding value in Y.

    Args:
        commutative (bool): determines if Deep Feature Synthesis should
            generate both x / y and y / x, or just one. If True, there is
            no guarantee which of the two will be generated. Defaults to False.
    """

    name = "divide_numeric"
    input_types = [(cs.numeric(), cs.numeric())]
    return_type = pl.NUMERIC_DTYPES

    description_template = "the result of {} divided by {}"

    def __init__(self, commutative=False):
        super().__init__()
        self.commutative = commutative

    def _apply(self, df: TFrame) -> pl.Expr:
        return pl.col(df.columns[0]) / pl.col(df.columns[1])

    def _get_name(self, cols: list[str]):
        return f"{cols[0]} / {cols[1]}"
