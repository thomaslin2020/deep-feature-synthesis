from typing import Union

import polars as pl
import polars.selectors as cs

from dfs.primitives.base import TransformPrimitive, TFrame


class AddNumericScalar(TransformPrimitive):
    """Adds a scalar to a column X in a DataFrame.

    Description:
        Given a column with numeric values and a scalar, add the given scalar to each value in the list.
    """

    name = "add_numeric_scalar"
    input_types = [cs.numeric()]
    return_type = pl.NUMERIC_DTYPES

    def __init__(self, value: Union[int, float] = 0):
        super().__init__()
        self.value = value
        self.description_template = "the sum of {{}} and {}".format(self.value)

    def _apply(self, df: TFrame) -> pl.Expr:
        return pl.col(df.columns[0]) + pl.lit(self.value)

    def _get_name(self, cols: list[str]):
        return f'{cols[0]} + {self.value}'
