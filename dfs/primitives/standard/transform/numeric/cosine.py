import polars as pl
import polars.selectors as cs

from dfs.primitives.base import TransformPrimitive, TFrame


class Cosine(TransformPrimitive):
    """Computes the cosine of a number."""

    name = "cosine"
    input_types = [cs.numeric()]
    return_type = pl.FLOAT_DTYPES
    description_template = "the cosine of {}"

    def _apply(self, df: TFrame) -> pl.Expr:
        return pl.col(df.columns[0]).cos()
