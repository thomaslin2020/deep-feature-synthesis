from abc import ABC

from dfs.primitives.base.primitive_base import PrimitiveBase


class AggregationPrimitive(PrimitiveBase, ABC):
    def _get_name(self, base_feature_names: list[str],
                  relationship_path_name: str, parent_dataframe_name: str,
                  where_str: str, use_prev_str: bool):
        name = "%s(%s.%s%s%s%s)" % (
            self.name.upper(),
            relationship_path_name,
            ", ".join(base_feature_names),
            where_str,
            use_prev_str,
            self.get_args(),
        )

        if self.num_output_features == 1:
            return name

        return [f"{name}[{i}]" for i in range(self.num_output_features)]
