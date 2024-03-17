import polars as pl
import polars.selectors as cs

from dfs.primitives.base import TransformPrimitive, TFrame


class And(TransformPrimitive):
    """Performs element-wise logical AND of two lists.

    Description:
        Given a list of booleans X and a list of booleans Y,
        determine whether each value in X is `True`, and
        whether its corresponding value in Y is also `True`.
    """

    name = "and"
    input_types = [(cs.boolean(), cs.boolean())]
    return_type = pl.Boolean
    commutative = True
    description_template = "whether {} and {} are true"

    def _apply(self, df: TFrame) -> pl.Expr:
        return pl.col(df.columns[0]) & pl.col(df.columns[1])

    def _get_name(self, cols: list[str]):
        return f"AND({cols[0]}, {cols[1]})"
