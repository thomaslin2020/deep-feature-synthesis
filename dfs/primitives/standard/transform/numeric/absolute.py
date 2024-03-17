import polars as pl
import polars.selectors as cs

from dfs.primitives.base import TransformPrimitive, TFrame


class Absolute(TransformPrimitive):
    """Computes the absolute value of a number."""

    name = "absolute"
    input_types = [cs.numeric()]
    return_type = pl.NUMERIC_DTYPES
    description_template = "the absolute value of {}"

    def _apply(self, df: TFrame) -> pl.Expr:
        return pl.col(df.columns[0]).abs()
