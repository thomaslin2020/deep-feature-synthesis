import polars as pl
import polars.selectors as cs

from dfs.primitives.base import TransformPrimitive, TFrame


class Sine(TransformPrimitive):
    """Computes the sine of a number."""

    name = "sine"
    input_types = [cs.numeric()]
    return_type = pl.FLOAT_DTYPES
    description_template = "the sine of {}"

    def _apply(self, df: TFrame) -> pl.Expr:
        return pl.col(df.columns[0]).sin()
