from typing import Union

import polars as pl
import polars.selectors as cs

from dfs.primitives.base import TransformPrimitive, TFrame


class DivideNumericScalar(TransformPrimitive):
    """Divides each element in the list by a scalar.

    Description:
        Given a list of numeric values and a scalar, divide
        each value in the list by the scalar.
    """

    name = "divide_numeric_scalar"
    input_types = [cs.numeric()]
    return_type = pl.NUMERIC_DTYPES

    def __init__(self, value: Union[int, float] = 1):
        super().__init__()
        self.value = value
        self.description_template = "the result of {{}} divided by {}".format(self.value)

    def _apply(self, df: TFrame) -> pl.Expr:
        return pl.col(df.columns[0]) / pl.lit(self.value)

    def _get_name(self, cols: list[str]):
        return "%s / %s" % (cols[0], str(self.value))
