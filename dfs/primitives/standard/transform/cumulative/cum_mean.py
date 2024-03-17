import polars as pl
import polars.selectors as cs

from dfs.primitives.base import TransformPrimitive, TFrame


class CumMean(TransformPrimitive):
    """Calculates the cumulative mean.

    Description:
        Given a list of values, return the cumulative mean
        (or running mean). There is no set window, so the
        mean at each point is calculated over all prior values.
        `NaN` values will return `NaN`, but in the window of a
        cumulative caluclation, they're treated as 0.
    """

    name = "cum_mean"
    input_types = [cs.numeric()]
    return_type = pl.NUMERIC_DTYPES
    use_full_dataframe = True
    description_template = "the cumulative mean of {}"

    def _apply(self, df: TFrame) -> pl.Expr:
        return pl.col(df.columns[0]).cum_sum() / pl.col(df.columns[0]).cum_count()
