import polars as pl
import polars.selectors as cs

from dfs.primitives.base import TransformPrimitive, TFrame


class SameAsPrevious(TransformPrimitive):
    """Determines if a value is equal to the previous value in a list.

    Description:
        Compares a value in a list to the previous value and returns True if
        the value is equal to the previous value or False otherwise. The
        first item in the output will always be False, since there is no previous
        element for the first element comparison.

        Any nan values in the input will be filled using either a forward-fill
        or backward-fill method, specified by the fill_method argument. The number
        of consecutive nan values that get filled can be limited with the limit
        argument. Any nan values left after filling will result in False being
        returned for any comparison involving the nan value.

    Args:
        fill_method (str): Method for filling gaps in series. Valid
        options are `backfill`, `bfill`, `pad`, `ffill`.
        `pad / ffill`: fill gap with last valid observation.
        `backfill / bfill`: fill gap with next valid observation.
        Default is `pad`.

        limit (int): The max number of consecutive NaN values in a gap that
            can be filled. Default is None.
    """

    name = "same_as_previous"
    input_types = [cs.numeric()]
    return_type = pl.Boolean

    def __init__(self, value, strategy=None, limit=None):
        if strategy not in {None, 'forward', 'backward', 'min', 'max', 'mean', 'zero', 'one'}:
            raise ValueError("Invalid fill_method")
        self.value = value
        self.strategy = strategy
        self.limit = limit

    def _apply(self, df: TFrame) -> pl.Expr:
        expr = pl.col(df.columns[0])
        if self.strategy is not None:
            expr = expr.fill_null(self.value, strategy=self.strategy, limit=self.limit)
        return expr.eq(expr.shift())
