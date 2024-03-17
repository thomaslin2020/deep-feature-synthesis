import polars as pl
import polars.selectors as cs

from dfs.primitives.base import TransformPrimitive, TFrame


class CumCount(TransformPrimitive):
    """Calculates the cumulative count.

    Description:
        Given a list of values, return the cumulative count
        (or running count). There is no set window, so the
        count at each point is calculated over all prior
        values. `NaN` values are counted.
    """

    name = "cum_count"
    input_types = [cs.all()]
    return_type = pl.INTEGER_DTYPES
    use_full_dataframe = True
    description_template = "the cumulative count of {}"

    def _apply(self, df: TFrame) -> pl.Expr:
        return pl.cum_count(pl.col(df.columns[0]))
