import polars as pl
import polars.selectors as cs

from dfs.primitives.base import TransformPrimitive, TFrame


class NotEqual(TransformPrimitive):
    """Determines if values in one list are not equal to another list.

    Description:
        Given a list of values X and a list of values Y, determine
        whether each value in X is not equal to each corresponding
        value in Y.
    """

    name = "not_equal"
    input_types = [(cs.all(), cs.all())]
    return_type = pl.Boolean
    commutative = True
    
    description_template = "whether {} does not equal {}"

    def _apply(self, df: TFrame) -> pl.Expr:
        return pl.col(df.columns[0]).ne(pl.col(df.columns[1]))

    def _get_name(self, cols: list[str]):
        return f"{cols[0]} != {cols[1]}"
