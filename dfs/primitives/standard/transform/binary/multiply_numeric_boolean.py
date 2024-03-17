import polars as pl
import polars.selectors as cs

from dfs.primitives.base import TransformPrimitive, TFrame


class MultiplyNumericBoolean(TransformPrimitive):
    """Performs element-wise multiplication of a numeric list with a boolean list.

    Description:
        Given a list of numeric values X and a list of
        boolean values Y, return the values in X where
        the corresponding value in Y is True.
    """

    name = "multiply_numeric_boolean"
    input_types = [(cs.numeric(), cs.boolean()),
                   (cs.boolean(), cs.numeric())]
    return_type = pl.NUMERIC_DTYPES

    commutative = True
    description_template = "the product of {} and {}"

    def _apply(self, df: TFrame) -> pl.Expr:
        if df.dtypes[0] == pl.Boolean:
            bools = df.columns[0]
            vals = df.columns[1]
        else:
            bools = df.columns[1]
            vals = df.columns[0]
        return pl.col(vals) * pl.col(bools).cast(pl.Int64)

    def _get_name(self, cols: list[str]):
        return f"{cols[0]} * {cols[1]}"
