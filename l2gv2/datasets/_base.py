from pathlib import Path

from tqdm import tqdm
import polars as pl
import torch
import networkx as nx
import raphtory as rp
import torch_geometric.data

DATA_PATH = Path(__file__).parent / "data"
DATASETS = [x.stem for x in DATA_PATH.glob("*") if x.is_dir()]
FORMATS = ["networkx", "tgeometric", "raphtory", "polars", "edge_list"]
EDGE_COLUMNS = {"source", "dest"}  # required columns

EdgeList = list[tuple[str, str]]


def is_graph_dataset(p: Path) -> bool:
    "Returns True if dataset represented by `p` is a valid dataset"
    return (p / (p.stem + "_edges.parquet")).exists()


class DataLoader:
    """Take a dataframe representing a (temporal) graph and provide
    methods for loading the data in different formats.
    """

    def __init__(self, dset: str):
        if is_graph_dataset(Path(dset)):
            self.path = Path(dset)
        elif is_graph_dataset(DATA_PATH / dset):
            self.path = DATA_PATH / dset
        else:
            raise FileNotFoundError(f"Dataset not found: {dset}")

        self.paths = {"edges": self.path / (self.path.stem + "_edges.parquet")}
        if (nodes_path := self.path / (self.path.stem + "_nodes.parquet")).exists():
            self.paths["nodes"] = nodes_path

        self._load_files()

    def _load_files(self):
        "Loads dataset into memory"

        self.edges = pl.read_parquet(self.paths["edges"])
        assert EDGE_COLUMNS <= set(
            self.edges.columns
        ), f"Required edge columns not found: {EDGE_COLUMNS}"
        self.temporal = "timestamp" in self.edges.columns
        if not self.temporal:
            self.edges = self.edges.with_columns(pl.lit(0).alias("timestamp"))
        self.datelist = self.edges.select("timestamp").to_series().unique()

        # Process nodes
        if self.paths.get("nodes"):
            self.nodes = pl.read_parquet(self.paths["nodes"])
            assert (
                "nodes" in self.nodes.columns
            ), "Required node columns not found: 'nodes'"
        else:
            self.nodes = (
                pl.concat(
                    [
                        self.edges.select(
                            pl.col("timestamp"), pl.col("source").alias("nodes")
                        ),
                        self.edges.select(
                            pl.col("timestamp"), pl.col("dest").alias("nodes")
                        ),
                    ]
                )
                .unique()
                .sort(by=["timestamp", "nodes"])
            )

        self.edge_features = [
            x
            for x in self.edges.columns
            if x not in ["timestamp", "label"] + sorted(EDGE_COLUMNS)
        ]
        self.node_features = [
            x for x in self.nodes.columns if x not in ["timestamp", "label", "nodes"]
        ]

    def get_dates(self) -> list[str]:
        return self.datelist.to_list()

    def get_edges(self) -> pl.DataFrame:
        return self.edges

    def get_nodes(self, ts: str | None = None) -> pl.DataFrame:
        if ts is None:
            return self.nodes
        else:
            return self.nodes.filter(pl.col("timestamp") == ts)

    def get_node_list(self, ts: str | None = None) -> list[str]:
        nodes = self.nodes
        if ts is not None:
            nodes = nodes.filter(pl.col("timestamp") == ts)
        return nodes.select("nodes").unique(maintain_order=True).to_series().to_list()

    def get_node_features(self) -> list[str]:
        return self.node_features

    def get_edge_features(self) -> list[str]:
        return self.edge_features

    def get_graph(self) -> rp.Graph:
        g = rp.Graph()
        g.load_edges_from_pandas(
            df=self.edges.to_pandas(),
            time="timestamp",
            src="source",
            dst="dest",
            properties=self.edge_features,
        )
        g.load_nodes_from_pandas(
            df=self.nodes.to_pandas(),
            time="timestamp",
            id="nodes",
            properties=self.node_features,
        )
        return g

    def get_edge_list(self, temp: bool = True) -> EdgeList | dict[str, EdgeList]:
        if self.temporal and temp:
            edge_list = {}
            for d in tqdm(self.datelist):
                edges = (
                    self.edges.filter(pl.col("timestamp") == d)
                    .select("source", "dest")
                    .to_numpy()
                )
                edge_list[d] = [tuple(x) for x in edges]
        else:
            edges = self.edges.select("source", "dest").unique().to_numpy()
            edge_list = [tuple(x) for x in edges]
        return edge_list

    def get_networkx(self, temp: bool = True) -> nx.Graph | dict[str, nx.Graph]:
        if self.temporal and temp:
            nx_graphs = {}
            for d in tqdm(self.datelist):
                edges = (
                    self.edges.filter(pl.col("timestamp") == d)
                    .select("source", "dest")
                    .to_numpy()
                )
                edge_list = [tuple(x) for x in edges]
                nx_graphs[d] = nx.from_edgelist(edge_list)
        else:
            edges = self.edges.select("source", "dest").unique().to_numpy()
            edge_list = [tuple(x) for x in edges]
            nx_graphs = nx.from_edgelist(edge_list)
        return nx_graphs

    def get_edge_index(
        self, temp: bool = True
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if self.temporal and temp:
            edge_index = {}
            for d in tqdm(self.datelist):
                edges = (
                    self.edges.filter(pl.col("timestamp") == d)
                    .select("source", "dest")
                    .to_numpy()
                )
                edge_list = [tuple(x) for x in edges]
                edge_index[d] = (
                    torch.tensor(edge_list, dtype=torch.long).t().contiguous()
                )
        else:
            edges = self.edges.select("source", "dest").unique().to_numpy()
            edge_list = [tuple(x) for x in edges]
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        return edge_index

    def get_tgeometric(
        self, temp: bool = True
    ) -> torch_geometric.data.Data | dict[str, torch_geometric.data.Data]:
        nodes = self.nodes.select("nodes").unique().to_numpy()
        features = self.nodes.select([c for c in self.node_features]).to_numpy()
        if self.temporal and temp:
            tg_graphs = {}
            for d in tqdm(self.datelist):
                edges = (
                    self.edges.filter(pl.col("timestamp") == d)
                    .select("source", "dest")
                    .to_numpy()
                )
                edge_list = [tuple(x) for x in edges]
                edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
                tg = torch_geometric.data.Data(edge_index=edge_index)
                tg.nodes = torch.from_numpy(nodes).int()
                tg.x = torch.from_numpy(features).float()
                tg_graphs[d] = tg
        else:
            edges = self.edges.select("source", "dest").unique().to_numpy()
            edge_list = [tuple(x) for x in edges]
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            tg_graphs = torch_geometric.data.Data(edge_index=edge_index)
            tg_graphs.nodes = torch.Tensor(nodes).int()
            tg_graphs.x = torch.from_numpy(features).float()
        return tg_graphs

    def summary(self):
        pass


# TODO: uncomment and integrate into DataLoader?
# def summary(data: nx.Graph | list[nx.Graph]):
#     if not isinstance(data, list):
#         data = [data]
#
#     if isinstance(data[0], nx.Graph):
#         number_of_nodes = []
#         number_of_edges = []
#         avg_degree = []
#         for G in data:
#             number_of_nodes.append(G.number_of_nodes())
#             number_of_edges.append(G.number_of_edges())
#
#             avg_degree.append(sum(dict(G.degree()).values()) / G.number_of_nodes())
#         avg_n = int(np.mean(number_of_nodes))
#         avg_e = int(np.mean(number_of_edges))
#         avg_d = int(np.mean(avg_degree))
#
#         print(
#             "The dataset consists of {} graphs.\n Average number of nodes : {} \n Average number of edges : {} \n Average degree: {} ".format(
#                 len(data), avg_n, avg_e, avg_d
#             )
#         )
#     else:
#         print("Converting into networkx graphs")
#         data = [to_networkx(d, to_undirected=False) for d in tqdm(data)]
#         summary(data)
