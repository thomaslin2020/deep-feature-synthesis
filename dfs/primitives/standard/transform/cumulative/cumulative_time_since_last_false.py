import polars as pl
import polars.selectors as cs

from dfs.primitives.base import TransformPrimitive, TFrame


class CumulativeTimeSinceLastFalse(TransformPrimitive):
    """Determines the time since last `False` value.

    Description:
        Given a list of booleans and a list of corresponding
        datetimes, determine the time at each point since the
        last `False` value. Returns time difference in seconds.
        `NaN` values are ignored.
    """

    name = "cumulative_time_since_last_false"
    input_types = [(cs.datetime() | cs.date(), cs.boolean())]
    return_type = pl.DURATION_DTYPES

    def _apply(self, df: TFrame) -> pl.Expr:
        return (
            pl.col(df.columns[0]).sub(
                pl.when(pl.col(df.columns[1]).eq(pl.lit(False)))
                .then(pl.col(df.columns[0])).otherwise(None)
                .fill_null(strategy='forward')
            )
        )

