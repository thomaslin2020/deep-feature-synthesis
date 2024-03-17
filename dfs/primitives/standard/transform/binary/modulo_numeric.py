import polars as pl
import polars.selectors as cs

from dfs.primitives.base import TransformPrimitive, TFrame


class ModuloNumeric(TransformPrimitive):
    """Performs element-wise modulo of two lists.

    Description:
        Given a list of values X and a list of values Y,
        determine the modulo, or remainder of each value in
        X after it's divided by its corresponding value in Y.
    """

    name = "modulo_numeric"
    input_types = [(cs.numeric(), cs.numeric())]
    return_type = pl.NUMERIC_DTYPES
    
    description_template = "the remainder after dividing {} by {}"

    def _apply(self, df: TFrame) -> pl.Expr:
        return pl.col(df.columns[0]) % pl.col(df.columns[1])

    def _get_name(self, cols: list[str]):
        return f"{cols[0]} % {cols[1]}"
