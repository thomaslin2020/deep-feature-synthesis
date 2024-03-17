import polars as pl
import polars.selectors as cs

from dfs.primitives.base import TransformPrimitive, TFrame


class ModuloByFeature(TransformPrimitive):
    """Computes the modulo of a scalar by each element in a list.

    Description:
        Given a list of numeric values and a scalar, return the
        modulo, or remainder of the scalar after being divided
        by each value.
    """

    name = "modulo_by_feature"
    input_types = [cs.numeric()]
    return_type = pl.NUMERIC_DTYPES

    def __init__(self, value=1):
        self.value = value
        self.description_template = "the remainder after dividing {} by {{}}".format(self.value)

    def _apply(self, df: TFrame) -> pl.Expr:
        return pl.lit(self.value) % pl.col(df.columns[0])

    def _get_name(self, cols: list[str]):
        return f"{str(self.value)} % {cols[0]}"
