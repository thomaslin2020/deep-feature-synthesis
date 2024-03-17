from typing import Union

import polars as pl
import polars.selectors as cs

from dfs.primitives.base import TransformPrimitive, TFrame


class ScalarSubtractNumericFeature(TransformPrimitive):
    """Subtracts each value in the list from a given scalar.

    Description:
        Given a list of numeric values and a scalar, subtract
        each value from the scalar and return the list of
        differences.
    """

    name = "scalar_subtract_numeric_feature"
    input_types = [cs.numeric()]
    return_type = pl.NUMERIC_DTYPES

    def __init__(self, value: Union[int, float] = 0):
        self.value = value
        self.description_template = "the result {} minus {{}}".format(self.value)

    def _apply(self, df: TFrame) -> pl.Expr:
        return pl.lit(self.value) - pl.col(df.columns[0])

    def _get_name(self, cols: list[str]):
        return f"{str(self.value)} - {cols[0]}"
