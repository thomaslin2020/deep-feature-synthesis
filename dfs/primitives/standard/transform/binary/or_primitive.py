import polars as pl
import polars.selectors as cs

from dfs.primitives.base import TransformPrimitive, TFrame


class Or(TransformPrimitive):
    """Performs element-wise logical OR of two lists.

    Description:
        Given a list of booleans X and a list of booleans Y,
        determine whether each value in X is `True`, or
        whether its corresponding value in Y is `True`.
    """

    name = "or"
    input_types = [(cs.boolean(), cs.boolean())]
    return_type = pl.Boolean
    commutative = True
    
    description_template = "whether {} is true or {} is true"

    def _apply(self, df: TFrame) -> pl.Expr:
        return pl.col(df.columns[0]) | pl.col(df.columns[1])

    def _get_name(self, cols: list[str]):
        return f"OR({cols[0]}, {cols[1]})"
