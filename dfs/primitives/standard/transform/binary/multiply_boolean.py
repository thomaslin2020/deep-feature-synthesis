import polars as pl
import polars.selectors as cs

from dfs.primitives.base import TransformPrimitive, TFrame


class MultiplyBoolean(TransformPrimitive):
    """Performs element-wise multiplication of two lists of boolean values.

    Description:
        Given a list of boolean values X and a list of boolean
        values Y, determine the product of each value in X
        with its corresponding value in Y.
    """

    name = "multiply_boolean"
    input_types = [(cs.boolean(), cs.boolean())]
    return_type = pl.Boolean
    commutative = True
    
    description_template = "the product of {} and {}"

    def _apply(self, df: TFrame) -> pl.Expr:
        return pl.col(df.columns[0]) & pl.col(df.columns[1])

    def _get_name(self, cols: list[str]):
        return f"{cols[0]} * {cols[1]}"
