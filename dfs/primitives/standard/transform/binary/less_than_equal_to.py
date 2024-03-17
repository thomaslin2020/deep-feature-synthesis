import polars as pl
import polars.selectors as cs

from dfs.primitives.base import TransformPrimitive, TFrame


class LessThanEqualTo(TransformPrimitive):
    """Determines if values in one list are less than or equal to another list.

    Description:
        Given a list of values X and a list of values Y, determine
        whether each value in X is less than or equal to each
        corresponding value in Y. Equal pairs will return `True`.
    """

    name = "less_than_equal_to"
    input_types = [(cs.numeric(), cs.numeric()),
                   (cs.datetime() | cs.date(), cs.datetime() | cs.date()),
                   (cs.duration(), cs.duration())]
    return_type = pl.Boolean
    
    description_template = "whether {} is less than or equal to {}"

    def _apply(self, df: TFrame) -> pl.Expr:
        return pl.col(df.columns[0]).le(pl.col(df.columns[1]))

    def _get_name(self, cols: list[str]):
        return f"{cols[0]} <= {cols[1]}"
