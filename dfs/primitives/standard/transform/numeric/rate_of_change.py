import polars as pl
import polars.selectors as cs

from dfs.primitives.base import TransformPrimitive, TFrame


class RateOfChange(TransformPrimitive):
    """Computes the rate of change of a value per second."""

    name = "rate_of_change"
    input_types = [(cs.numeric(), cs.datetime() | cs.date())]
    return_type = pl.FLOAT_DTYPES
    use_full_dataframe = True
    description_template = "the rate of change of {} per second"

    def _apply(self, df: TFrame) -> pl.Expr:
        return pl.col(df.columns[0]) / pl.col(df.columns[1]).diff().dt.total_seconds()
