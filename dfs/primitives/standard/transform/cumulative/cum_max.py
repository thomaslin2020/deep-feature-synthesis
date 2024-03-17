import polars as pl
import polars.selectors as cs

from dfs.primitives.base import TransformPrimitive, TFrame


class CumMax(TransformPrimitive):
    """Calculates the cumulative maximum.

    Description:
        Given a list of values, return the cumulative max
        (or running max). There is no set window, so the max
        at each point is calculated over all prior values.
        `NaN` values will return `NaN`, but in the window of a
        cumulative calculation, they're ignored.
    """

    name = "cum_max"
    input_types = [cs.numeric()]
    return_type = pl.NUMERIC_DTYPES
    use_full_dataframe = True
    description_template = "the cumulative maximum of {}"

    def _apply(self, df: TFrame) -> pl.Expr:
        return pl.col(df.columns[0]).cum_max()
