from typing import Union

import polars as pl
import polars.selectors as cs

from dfs.primitives.base import TransformPrimitive, TFrame


class ModuloNumericScalar(TransformPrimitive):
    """Computes the modulo of each element in the list by a given scalar.

    Description:
        Given a list of numeric values and a scalar, return
        the modulo, or remainder of each value after being
        divided by the scalar.
    """

    name = "modulo_numeric_scalar"
    input_types = [cs.numeric()]
    return_type = pl.NUMERIC_DTYPES

    def __init__(self, value: Union[int, float] = 1):
        self.value = value
        self.description_template = "the remainder after dividing {{}} by {}".format(self.value)

    def _apply(self, df: TFrame) -> pl.Expr:
        return pl.col(df.columns[0]) % pl.lit(self.value)

    def _get_name(self, cols: list[str]):
        return f"{cols[0]} % {str(self.value)}"
