import polars as pl
import polars.selectors as cs

from dfs.primitives.base import TransformPrimitive, TFrame


class Tangent(TransformPrimitive):
    """Computes the tangent of a number."""

    name = "tangent"
    input_types = [cs.numeric()]
    return_type = pl.FLOAT_DTYPES
    description_template = "the tangent of {}"

    def _apply(self, df: TFrame) -> pl.Expr:
        return pl.col(df.columns[0]).tan()
