import polars as pl
import polars.selectors as cs

from dfs.primitives.base import TransformPrimitive, TFrame


class SquareRoot(TransformPrimitive):
    """Computes the square root of a number."""

    name = "square_root"
    input_types = [cs.numeric()]
    return_type = pl.FLOAT_DTYPES
    description_template = "the square root of {}"

    def _apply(self, df: TFrame) -> pl.Expr:
        return pl.col(df.columns[0]).sqrt()
