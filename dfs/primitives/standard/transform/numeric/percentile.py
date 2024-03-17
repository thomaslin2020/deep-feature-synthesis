import polars as pl
import polars.selectors as cs

from dfs.primitives.base import TransformPrimitive, TFrame


class Percentile(TransformPrimitive):
    """Determines the percentile rank for each value in a list."""

    name = "percentile"
    use_full_dataframe = True
    input_types = [cs.numeric()]
    return_type = pl.NUMERIC_DTYPES
    description_template = "the percentile rank of {}"

    def _apply(self, df: TFrame) -> pl.Expr:
        return pl.col(df.columns[0]).rank() / pl.col(df.columns[0]).count()
