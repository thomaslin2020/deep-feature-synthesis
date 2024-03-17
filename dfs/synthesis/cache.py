import polars as pl
from anytree import Node, RenderTree


class FeatureCache:
    def __init__(self, df: pl.LazyFrame):
        self.root = Node("root")
        self.cache = {}
        self.finalized = False

        for col in df.columns:
            self.cache[col] = Node(col, parent=self.root)

    def clear(self):
        self.cache = {}
        self.root = Node("root")
        self.finalized = False

    def render(self):
        print(RenderTree(self.root))
