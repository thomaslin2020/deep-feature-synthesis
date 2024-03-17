import polars as pl
import polars.selectors as cs

from dfs.primitives.base import TransformPrimitive, TFrame


class Equal(TransformPrimitive):
    """Determines if values in one list are equal to another list.

    Description:
        Given a list of values X and a list of values Y, determine
        whether each value in X is equal to each corresponding value
        in Y.
    """

    name = "equal"
    input_types = [(cs.all(), cs.all())]
    return_type = pl.Boolean
    commutative = True

    description_template = "whether {} equals {}"

    def _apply(self, df: TFrame) -> pl.Expr:
        return pl.col(df.columns[0]).eq(pl.col(df.columns[1]))

    def _get_name(self, cols: list[str]):
        return f"{cols[0]} = {cols[1]}"
