import polars as pl
import polars.selectors as cs

from dfs.primitives.base import TransformPrimitive, TFrame


class CumulativeTimeSinceLastTrue(TransformPrimitive):
    """Determines the time (in seconds) since the last boolean was `True`
    given a datetime index column and boolean column
    """

    name = "cumulative_time_since_last_true"
    input_types = [(cs.datetime() | cs.date(), cs.boolean())]
    return_type = pl.DURATION_DTYPES

    def _apply(self, df: TFrame) -> pl.Expr:
        return (
            pl.col(df.columns[0]).sub(
                pl.when(pl.col(df.columns[1]).eq(pl.lit(True)))
                .then(pl.col(df.columns[0])).otherwise(None)
                .fill_null(strategy='forward')
            )
        )
