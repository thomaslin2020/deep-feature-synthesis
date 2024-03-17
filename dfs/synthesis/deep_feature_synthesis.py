import random
import re
from typing import Union, Optional

import polars as pl

from dfs import primitives
from dfs.primitives import TransformPrimitive
from dfs.synthesis.cache import FeatureCache

camel_case_pattern = re.compile(r'(?<!^)(?=[A-Z])')


def _to_snake_case(name: str):
    name = ''.join(list(filter(lambda x: x, map(lambda x: x.strip().capitalize(), name.split(' ')))))
    return camel_case_pattern.sub('_', name).lower().strip()


def check_transform_primitive(primitive, is_group: bool, transform_primitive_dict):
    prim_dict = transform_primitive_dict
    supertype = TransformPrimitive
    arg_name = "trans_primitives" if not is_group else "group_by_trans_primitives"

    if isinstance(primitive, str):
        prim_string = _to_snake_case(primitive)
        if prim_string not in prim_dict:
            raise ValueError(
                f"Unknown transform primitive {prim_string}. "
                "Call ft.primitives.list_primitives() to get a list of available primitives"
            )
        primitive = prim_dict[prim_string]
    if isinstance(primitive, type):
        primitive = primitive()
    if not isinstance(primitive, supertype):
        raise ValueError(f"Primitive {type(primitive)} in {arg_name} is not a transform primitive")
    return primitive


class DeepFeatureSynthesis:
    """Automatically produce features for a dataframe.

    Args:
        dataframe (Union[pl.DataFrame, pl.LazyFrame]): Dataframe for which to build features.

        trans_primitives (list[str or :class:`.primitives.TransformPrimitive`], optional):
            list of Transform primitives to use.

            Default: ["day", "year", "month", "weekday"]

        max_depth (int, optional) : maximum allowed depth of features.
            Default: 2. If -1, no limit.

        max_features (int, optional) : Cap the number of generated features to
            this number. If -1, no limit.

        ignore_columns (list[str], optional): List of specific
            columns within each dataframe to blacklist when creating features.
            If None, use all columns.

        primitive_options (dict[str or tuple[str] or PrimitiveBase -> dict or list[dict]], optional):
            Specify options for a single primitive or a group of primitives.
            Lists of option dicts are used to specify options per input for primitives
            with multiple inputs. Each option ``dict`` can have the following keys:

            ``"include_columns"``
                List of specific columns within each dataframe to include when
                creating features for the primitive(s). All other columns
                in a given dataframe will be ignored (list[str]).
            ``"ignore_columns"``
                List of specific columns within each dataframe to blacklist
                when creating features for the primitive(s) (list[str]).
            ``"include_group_by_columns"``
                List of specific columns within each dataframe to include as
                group_bys, if applicable. All other columns in each
                dataframe will be ignored (list[str]).
            ``"ignore_group_by_columns"``
                List of specific columns within each dataframe to blacklist
                as group_bys (list[str]).
    """

    def __init__(
            self, dataframe: Union[pl.DataFrame, pl.LazyFrame], group_cols: list[str] = None,
            trans_primitives: list = None, max_depth: Optional[int] = 2,
            max_features: Optional[int] = None, ignore_columns: list = None,
            primitive_options: dict = None
    ):
        self.max_depth = max_depth or 0
        self.max_features = max_features or 10
        for col in ignore_columns:
            if col not in dataframe.columns:
                raise ValueError(f"Column {col} not found in dataframe")

        self.group_cols = group_cols or []

        self.dataframe: pl.LazyFrame = dataframe.lazy().drop(pl.col(col) for col in (ignore_columns or []))
        self.cache = FeatureCache(self.dataframe)

        transform_primitive_dict = primitives.get_transform_primitives()

        if trans_primitives is None:
            trans_primitives = primitives.get_default_transform_primitives()

        self.trans_primitives = sorted([
            check_transform_primitive(p, False, transform_primitive_dict) for p in trans_primitives
        ])

    def _select_columns(self, input_type, df: pl.LazyFrame) -> list[str]:
        if isinstance(input_type, list):
            return self._select_columns(random.choice(input_type), df)

        if isinstance(input_type, tuple):
            cols = []
            for param_type in input_type:
                choices = df.select(param_type).columns
                if not choices:
                    raise ValueError(f"No columns of type {param_type} found in DataFrame")
                col = random.choice(choices)
                cols.append(col)
                df = df.drop(col)
            return cols
        else:
            return random.choice(df.select(input_type).columns)

    def _build_features(self):
        # Get random feature

        trans_primitive: TransformPrimitive = random.choice(self.trans_primitives)
        input_types = trans_primitive.input_types

        assert len(input_types) > 0, "Primitive must have at least one input type"
        subset = self._select_columns(input_types, self.dataframe.drop(self.group_cols))

        self.dataframe = trans_primitive.apply(self.dataframe, subset, self.group_cols)

    def build(self):
        self.cache.clear()

        # Doesn't take into account of max_depth yet
        for _ in range(self.max_features):
            self._build_features()
        self.cache.finalized = True

    def render(self):
        if not self.cache.finalized:
            raise ValueError("Features have not been built yet. Call `build` to build features")
        self.cache.render()

    def run(self) -> pl.DataFrame:
        if not self.cache.finalized:
            raise ValueError("Features have not been built yet. Call `build` to build features")

        return self.dataframe.collect()
