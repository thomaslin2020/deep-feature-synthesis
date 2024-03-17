import polars as pl
import polars.selectors as cs

from dfs.primitives.base import TransformPrimitive, TFrame


class Negate(TransformPrimitive):
    """Negates a numeric value."""

    name = "negate"
    input_types = [cs.numeric()]
    return_type = pl.NUMERIC_DTYPES
    description_template = "the negation of {}"

    def _apply(self, df: TFrame) -> pl.Expr:
        return pl.col(df.columns[0]).neg()

    def _get_name(self, cols: list[str]):
        return f"-({cols[0]})"
