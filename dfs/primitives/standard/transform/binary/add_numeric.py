import polars as pl
import polars.selectors as cs

from dfs.primitives.base import TransformPrimitive, TFrame


class AddNumeric(TransformPrimitive):
    """Performs element-wise addition of two columns.

    Description:
        Given a column X and a column Y, return X + Y.
    """

    name = "add_numeric"
    input_types = [(cs.numeric(), cs.numeric())]
    return_type = pl.NUMERIC_DTYPES
    commutative = True
    description_template = "the sum of {} and {}"

    def _apply(self, df: TFrame) -> pl.Expr:
        return pl.col(df.columns[0]) + pl.col(df.columns[1])

    def _get_name(self, cols: list[str]):
        left, right = cols
        return f"{left} + {right}"
