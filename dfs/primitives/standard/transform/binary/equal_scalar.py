import polars as pl
import polars.selectors as cs

from dfs.primitives.base import TransformPrimitive, TFrame


class EqualScalar(TransformPrimitive):
    """Determines if values in a list are equal to a given scalar.

    Description:
        Given a list of values and a constant scalar, determine
        whether each of the values is equal to the scalar.
    """

    name = "equal_scalar"
    input_types = [cs.all()]
    return_type = pl.Boolean

    def __init__(self, value=None):
        super().__init__()
        self.value = value
        self.description_template = "whether {{}} equals {}".format(self.value)

    def _apply(self, df: TFrame) -> pl.Expr:
        return pl.col(df.columns[0]).eq(pl.lit(self.value))

    def _get_name(self, cols: list[str]):
        return f"{cols[0]} = {str(self.value)}"
