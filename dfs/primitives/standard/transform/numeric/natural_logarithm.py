import polars as pl
import polars.selectors as cs

from dfs.primitives.base import TransformPrimitive, TFrame


class NaturalLogarithm(TransformPrimitive):
    """Computes the natural logarithm of a number."""

    name = "natural_logarithm"
    input_types = [cs.numeric()]
    return_type = pl.FLOAT_DTYPES
    description_template = "the natural logarithm of {}"

    def _apply(self, df: TFrame) -> pl.Expr:
        return pl.col(df.columns[0]).ln()
