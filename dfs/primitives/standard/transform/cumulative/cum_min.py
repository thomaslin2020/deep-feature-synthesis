import polars as pl
import polars.selectors as cs

from dfs.primitives.base import TransformPrimitive, TFrame


class CumMin(TransformPrimitive):
    """Calculates the cumulative minimum.

    Description:
        Given a list of values, return the cumulative min
        (or running min). There is no set window, so the min
        at each point is calculated over all prior values.
        `NaN` values will return `NaN`, but in the window of a
        cumulative caluclation, they're ignored.
    """

    name = "cum_min"
    input_types = [cs.numeric()]
    return_type = pl.NUMERIC_DTYPES
    use_full_dataframe = True
    description_template = "the cumulative minimum of {}"

    def _apply(self, df: TFrame) -> pl.Expr:
        return pl.col(df.columns[0]).cum_min()
