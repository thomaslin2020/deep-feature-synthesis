from typing import Union

import polars as pl
import polars.selectors as cs

from dfs.primitives.base import TransformPrimitive, TFrame


class DivideByFeature(TransformPrimitive):
    """Divides a scalar by each value in the list.

    Description:
        Given a list of numeric values and a scalar, divide
        the scalar by each value and return the list of
        quotients.
    """

    name = "divide_by_feature"
    input_types = [cs.numeric()]
    return_type = pl.NUMERIC_DTYPES

    def __init__(self, value: Union[int, float] = 1):
        super().__init__()
        self.value = value
        self.description_template = "the result of {} divided by {{}}".format(self.value)

    def _apply(self, df: TFrame) -> pl.Expr:
        return pl.lit(self.value) / pl.col(df.columns[0])

    def _get_name(self, cols: list[str]):
        return f'{self.value} / {cols[0]}'
