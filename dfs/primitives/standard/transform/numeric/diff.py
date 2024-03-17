import polars as pl
import polars.selectors as cs

from dfs.primitives.base import TransformPrimitive, TFrame


class Diff(TransformPrimitive):
    """Computes the difference between the value in a list and the
    previous value in that list.

    Args:
        periods (int): The number of periods by which to shift the index row.
            Default is 0. Periods correspond to rows.

    Description:
        Given a list of values, compute the difference from the previous
        item in the list. The result for the first element of the list will
        always be `NaN`.
    """

    name = "diff"
    input_types = [cs.numeric()]
    return_type = pl.NUMERIC_DTYPES
    use_full_dataframe = True
    description_template = "the difference from the previous value of {}"

    def __init__(self, periods: int = 0):
        self.periods = periods

    def _apply(self, df: TFrame) -> pl.Expr:
        return pl.col(df.columns[0]).diff(self.periods)
