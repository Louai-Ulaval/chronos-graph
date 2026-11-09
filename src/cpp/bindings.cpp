// pybind11 bindings for the Chronos-Graph engine (module: chronos_cpp).
//
// Heavy calls release the GIL; the engine is internally synchronised.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "graph_engine.hpp"

namespace py = pybind11;
using namespace chronos;

namespace {

using U32Array = py::array_t<std::uint32_t, py::array::c_style | py::array::forcecast>;

std::span<const std::uint32_t> as_span(const U32Array& a) {
  if (a.ndim() != 1) throw std::invalid_argument("expected a 1-D array");
  return {a.data(), static_cast<std::size_t>(a.size())};
}

py::array_t<double> to_numpy(std::vector<double>&& v) {
  auto* heap = new std::vector<double>(std::move(v));
  py::capsule owner(heap, [](void* p) { delete static_cast<std::vector<double>*>(p); });
  return py::array_t<double>({heap->size()}, {sizeof(double)}, heap->data(), owner);
}

py::dict hop_dict(const std::vector<HopResult>& hops) {
  py::dict out;
  for (const HopResult& h : hops) out[py::int_(h.node)] = py::int_(h.distance);
  return out;
}

const char* edge_kind_name(EdgeKind k) {
  switch (k) {
    case EdgeKind::Call: return "call";
    case EdgeKind::Import: return "import";
    case EdgeKind::Contains: return "contains";
    case EdgeKind::Inherits: return "inherits";
    case EdgeKind::Reference: return "reference";
  }
  return "unknown";
}

}  // namespace

PYBIND11_MODULE(chronos_cpp, m) {
  m.doc() = "Chronos-Graph C++20 graph engine: CSR call/dependency graph, k-hop BFS, "
            "bidirectional shortest path, PageRank and token-budget subgraph pruning.";
  m.attr("__version__") = "1.0.0";
  m.attr("CPLUSPLUS") = static_cast<long>(__cplusplus);
#if defined(__aarch64__) || defined(_M_ARM64)
  m.attr("ARCH") = "arm64";
#elif defined(__x86_64__) || defined(_M_X64)
  m.attr("ARCH") = "x86_64";
#else
  m.attr("ARCH") = "unknown";
#endif

  py::enum_<EdgeKind>(m, "EdgeKind")
      .value("CALL", EdgeKind::Call)
      .value("IMPORT", EdgeKind::Import)
      .value("CONTAINS", EdgeKind::Contains)
      .value("INHERITS", EdgeKind::Inherits)
      .value("REFERENCE", EdgeKind::Reference);

  py::enum_<Direction>(m, "Direction")
      .value("OUT", Direction::Out)
      .value("IN", Direction::In)
      .value("BOTH", Direction::Both);

  py::class_<NodeInfo>(m, "NodeInfo")
      .def_readonly("name", &NodeInfo::name)
      .def_readonly("kind", &NodeInfo::kind)
      .def_readonly("file", &NodeInfo::file)
      .def_readonly("line", &NodeInfo::line)
      .def_readonly("token_cost", &NodeInfo::token_cost)
      .def("__repr__", [](const NodeInfo& n) {
        return "<NodeInfo " + n.name + " (" + n.kind + ") " + n.file + ":" +
               std::to_string(n.line) + ">";
      });

  py::class_<Edge>(m, "Edge")
      .def_readonly("src", &Edge::src)
      .def_readonly("dst", &Edge::dst)
      .def_readonly("kind", &Edge::kind)
      .def("__repr__", [](const Edge& e) {
        return "<Edge " + std::to_string(e.src) + " -" + edge_kind_name(e.kind) + "-> " +
               std::to_string(e.dst) + ">";
      });

  py::class_<PruneOptions>(m, "PruneOptions")
      .def(py::init<>())
      .def_readwrite("max_hops", &PruneOptions::max_hops)
      .def_readwrite("token_budget", &PruneOptions::token_budget)
      .def_readwrite("hop_decay", &PruneOptions::hop_decay)
      .def_readwrite("centrality_weight", &PruneOptions::centrality_weight)
      .def_readwrite("cost_exponent", &PruneOptions::cost_exponent)
      .def_readwrite("seed_floor", &PruneOptions::seed_floor)
      .def_readwrite("direction", &PruneOptions::direction)
      .def_readwrite("max_nodes", &PruneOptions::max_nodes)
      .def_property(
          "kind_weight",
          [](const PruneOptions& o) {
            return std::vector<double>(o.kind_weight.begin(), o.kind_weight.end());
          },
          [](PruneOptions& o, const std::vector<double>& w) {
            if (w.size() != kEdgeKindCount) {
              throw std::invalid_argument("kind_weight needs one weight per EdgeKind");
            }
            std::copy(w.begin(), w.end(), o.kind_weight.begin());
          });

  py::class_<PrunedNode>(m, "PrunedNode")
      .def_readonly("node", &PrunedNode::node)
      .def_readonly("distance", &PrunedNode::distance)
      .def_readonly("relevance", &PrunedNode::relevance)
      .def_readonly("centrality", &PrunedNode::centrality)
      .def_readonly("score", &PrunedNode::score)
      .def_readonly("token_cost", &PrunedNode::token_cost)
      .def_readonly("seed", &PrunedNode::seed);

  py::class_<PruneResult>(m, "PruneResult")
      .def_readonly("nodes", &PruneResult::nodes)
      .def_readonly("edges", &PruneResult::edges)
      .def_readonly("tokens_used", &PruneResult::tokens_used)
      .def_readonly("token_budget", &PruneResult::token_budget)
      .def_readonly("candidates", &PruneResult::candidates);

  py::class_<GraphEngine>(m, "GraphEngine")
      .def(py::init<>())
      .def("add_node", &GraphEngine::add_node, py::arg("name"), py::arg("kind") = "symbol",
           py::arg("file") = "", py::arg("line") = 0, py::arg("token_cost") = 1,
           "Insert a node (or update its metadata) and return its id.")
      .def("add_edge", &GraphEngine::add_edge, py::arg("src"), py::arg("dst"),
           py::arg("kind") = EdgeKind::Call)
      .def("add_edge_by_name", &GraphEngine::add_edge_by_name, py::arg("src"), py::arg("dst"),
           py::arg("kind") = EdgeKind::Call)
      .def(
          "add_edges",
          [](GraphEngine& g, const U32Array& src, const U32Array& dst, EdgeKind kind) {
            auto s = as_span(src);
            auto d = as_span(dst);
            py::gil_scoped_release release;
            g.add_edges(s, d, kind);
          },
          py::arg("src"), py::arg("dst"), py::arg("kind") = EdgeKind::Call,
          "Bulk-insert edges from two equal-length uint32 arrays.")
      .def("add_anonymous_nodes", &GraphEngine::add_anonymous_nodes, py::arg("count"),
           py::call_guard<py::gil_scoped_release>())
      .def("reserve", &GraphEngine::reserve, py::arg("nodes"), py::arg("edges"))
      .def("set_token_cost", &GraphEngine::set_token_cost, py::arg("node"), py::arg("cost"))
      .def("finalize", &GraphEngine::finalize, py::call_guard<py::gil_scoped_release>())
      .def("clear", &GraphEngine::clear)
      .def_property_readonly("num_nodes", &GraphEngine::num_nodes)
      .def_property_readonly("num_edges", &GraphEngine::num_edges)
      .def("__len__", &GraphEngine::num_nodes)
      .def("find", &GraphEngine::find, py::arg("name"))
      .def("node", &GraphEngine::node, py::arg("node"))
      .def("neighbors", &GraphEngine::neighbors, py::arg("node"),
           py::arg("direction") = Direction::Out)
      .def("edges", &GraphEngine::edges)
      .def(
          "edge_arrays",
          [](GraphEngine& g) {
            std::vector<Edge> es;
            {
              py::gil_scoped_release release;
              es = g.edges();
            }
            py::array_t<std::uint32_t> src(es.size()), dst(es.size());
            py::array_t<std::uint8_t> kind(es.size());
            auto s = src.mutable_unchecked<1>();
            auto d = dst.mutable_unchecked<1>();
            auto k = kind.mutable_unchecked<1>();
            for (std::size_t i = 0; i < es.size(); ++i) {
              s(i) = es[i].src;
              d(i) = es[i].dst;
              k(i) = static_cast<std::uint8_t>(es[i].kind);
            }
            return py::make_tuple(src, dst, kind);
          },
          "Return (src, dst, kind) numpy arrays for every deduplicated edge.")
      .def(
          "degrees",
          [](GraphEngine& g, Direction dir) {
            std::vector<std::uint32_t> deg;
            {
              py::gil_scoped_release release;
              deg = g.degrees(dir);
            }
            return py::array_t<std::uint32_t>(deg.size(), deg.data());
          },
          py::arg("direction") = Direction::Both)
      .def("stats",
           [](GraphEngine& g) {
             GraphStats s;
             {
               py::gil_scoped_release release;
               s = g.stats();
             }
             py::dict by_kind;
             for (std::size_t i = 0; i < kEdgeKindCount; ++i) {
               by_kind[edge_kind_name(static_cast<EdgeKind>(i))] = s.edges_by_kind[i];
             }
             py::dict d;
             d["nodes"] = s.nodes;
             d["edges"] = s.edges;
             d["memory_bytes"] = s.memory_bytes;
             d["edges_by_kind"] = by_kind;
             d["max_out_degree"] = s.max_out_degree;
             d["max_in_degree"] = s.max_in_degree;
             return d;
           })
      .def(
          "k_hop",
          [](GraphEngine& g, const std::vector<NodeId>& seeds, std::uint32_t k, Direction dir) {
            std::vector<HopResult> hops;
            {
              py::gil_scoped_release release;
              hops = g.k_hop(seeds, k, dir);
            }
            return hop_dict(hops);
          },
          py::arg("seeds"), py::arg("k") = 2, py::arg("direction") = Direction::Both,
          "Multi-source BFS. Returns {node_id: hop_distance}.")
      .def(
          "k_hop_arrays",
          [](GraphEngine& g, const std::vector<NodeId>& seeds, std::uint32_t k, Direction dir) {
            std::vector<HopResult> hops;
            {
              py::gil_scoped_release release;
              hops = g.k_hop(seeds, k, dir);
            }
            py::array_t<std::uint32_t> nodes(hops.size()), dist(hops.size());
            auto n = nodes.mutable_unchecked<1>();
            auto d = dist.mutable_unchecked<1>();
            for (std::size_t i = 0; i < hops.size(); ++i) {
              n(i) = hops[i].node;
              d(i) = hops[i].distance;
            }
            return py::make_tuple(nodes, dist);
          },
          py::arg("seeds"), py::arg("k") = 2, py::arg("direction") = Direction::Both,
          "Multi-source BFS returning (node_ids, hop_distances) numpy arrays in BFS order.")
      .def(
          "k_hop_count",
          [](GraphEngine& g, const std::vector<NodeId>& seeds, std::uint32_t k, Direction dir) {
            py::gil_scoped_release release;
            return g.k_hop(seeds, k, dir).size();
          },
          py::arg("seeds"), py::arg("k") = 2, py::arg("direction") = Direction::Both,
          "Multi-source BFS returning only the size of the neighbourhood.")
      .def("shortest_path", &GraphEngine::shortest_path, py::arg("src"), py::arg("dst"),
           py::arg("direction") = Direction::Out, py::call_guard<py::gil_scoped_release>(),
           "Bidirectional BFS shortest path; empty list when unreachable.")
      .def(
          "pagerank",
          [](GraphEngine& g, double damping, std::uint32_t max_iter, double tol) {
            std::vector<double> pr;
            {
              py::gil_scoped_release release;
              pr = g.pagerank(damping, max_iter, tol);
            }
            return to_numpy(std::move(pr));
          },
          py::arg("damping") = 0.85, py::arg("max_iter") = 100, py::arg("tol") = 1e-10)
      .def(
          "personalized_pagerank",
          [](GraphEngine& g, const std::vector<NodeId>& seeds, const std::vector<double>& weights,
             double damping, std::uint32_t max_iter, double tol) {
            std::vector<double> pr;
            {
              py::gil_scoped_release release;
              pr = g.personalized_pagerank(seeds, weights, damping, max_iter, tol);
            }
            return to_numpy(std::move(pr));
          },
          py::arg("seeds"), py::arg("weights") = std::vector<double>{},
          py::arg("damping") = 0.85, py::arg("max_iter") = 100, py::arg("tol") = 1e-10)
      .def(
          "prune",
          [](GraphEngine& g, const std::vector<NodeId>& seeds,
             const std::vector<double>& seed_scores, const PruneOptions& opts) {
            py::gil_scoped_release release;
            return g.prune(seeds, seed_scores, opts);
          },
          py::arg("seeds"), py::arg("seed_scores") = std::vector<double>{},
          py::arg("options") = PruneOptions{},
          "Token-budget subgraph pruning around retrieval seeds.");
}
