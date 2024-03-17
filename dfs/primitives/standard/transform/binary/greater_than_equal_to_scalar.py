from typing import Union

import polars as pl
import polars.selectors as cs

from dfs.primitives.base import TransformPrimitive, TFrame


class GreaterThanEqualToScalar(TransformPrimitive):
    """Determines if values are greater than or equal to a given scalar.

    Description:
        Given a list of values and a constant scalar, determine
        whether each of the values is greater than or equal to the
        scalar. If a value is equal to the scalar, return `True`.
    """

    name = "greater_than_equal_to_scalar"
    input_types = [cs.numeric()]
    return_type = pl.Boolean

    def __init__(self, value: Union[int, float] = 0):
        self.value = value
        self.description_template = ("whether {{}} is greater than or equal to {}".format(self.value))

    def _apply(self, df: TFrame) -> pl.Expr:
        return pl.col(df.columns[0]) >= pl.lit(self.value)

    def _get_name(self, cols: list[str]):
        return f"{cols[0]} >= {self.value}"
