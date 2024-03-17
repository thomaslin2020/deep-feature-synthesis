import polars as pl
import polars.selectors as cs

from dfs.primitives.base import TransformPrimitive, TFrame


class NotEqualScalar(TransformPrimitive):
    """Determines if values in a list are not equal to a given scalar.

    Description:
        Given a list of values and a constant scalar, determine
        whether each of the values is not equal to the scalar.
    """

    name = "not_equal_scalar"
    input_types = [cs.all()]
    return_type = pl.Boolean
    
    def __init__(self, value=None):
        self.value = value
        self.description_template = "whether {{}} does not equal {}".format(self.value)

    def _apply(self, df: TFrame) -> pl.Expr:
        return pl.col(df.columns[0]).ne(pl.lit(self.value))

    def _get_name(self, cols: list[str]):
        return f"{cols[0]} != {str(self.value)}"
