import polars as pl
import polars.selectors as cs

from dfs.primitives.base import TransformPrimitive, TFrame


class MultiplyNumeric(TransformPrimitive):
    """Performs element-wise multiplication of two lists.

    Description:
        Given a list of values X and a list of values
        Y, determine the product of each value in X
        with its corresponding value in Y.
    """

    name = "multiply_numeric"
    input_types = [(cs.numeric(), cs.numeric())]
    return_type = pl.NUMERIC_DTYPES
    commutative = True
    
    description_template = "the product of {} and {}"

    def _apply(self, df: TFrame) -> pl.Expr:
        return pl.col(df.columns[0]) * pl.col(df.columns[1])

    def _get_name(self, cols: list[str]):
        return f"{cols[0]} * {cols[1]}"
