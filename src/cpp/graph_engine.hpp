// Chronos-Graph: contiguous-memory directed call & dependency graph.
//
// Storage model
//   * Nodes live in a dense vector indexed by NodeId (uint32).
//   * Edges are appended to a staging list, then compacted into two CSR
//     (compressed sparse row) arrays on first query: one for out-edges and one
//     for in-edges. Every traversal therefore walks contiguous memory.
//   * Scratch buffers for traversals use generation stamps, so a BFS never has
//     to clear an O(N) visited array between queries.
//
// All public methods are internally synchronised, which lets the Python
// bindings release the GIL during heavy work.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace chronos {

using NodeId = std::uint32_t;
inline constexpr NodeId kInvalidNode = 0xFFFFFFFFu;

enum class EdgeKind : std::uint8_t {
  Call = 0,
  Import = 1,
  Contains = 2,
  Inherits = 3,
  Reference = 4,
};
inline constexpr std::size_t kEdgeKindCount = 5;

enum class Direction : std::uint8_t { Out = 0, In = 1, Both = 2 };

struct NodeInfo {
  std::string name;
  std::string kind;
  std::string file;
  std::int32_t line = 0;
  std::uint32_t token_cost = 1;
};

struct Edge {
  NodeId src = kInvalidNode;
  NodeId dst = kInvalidNode;
  EdgeKind kind = EdgeKind::Call;
};

struct HopResult {
  NodeId node = kInvalidNode;
  std::uint32_t distance = 0;
};

struct PruneOptions {
  std::uint32_t max_hops = 2;
  std::uint64_t token_budget = 4000;
  double hop_decay = 0.6;           // relevance multiplier per hop
  double centrality_weight = 0.25;  // blend between propagated relevance and PageRank
  double cost_exponent = 0.35;      // 0 = pure score greedy, 1 = pure density greedy
  double seed_floor = 0.7;          // seeds at or above this relative score are always kept first
  Direction direction = Direction::Both;
  std::uint32_t max_nodes = 0;      // 0 = unlimited
  // Per-edge-kind attenuation, indexed by EdgeKind.
  std::array<double, kEdgeKindCount> kind_weight{1.0, 0.6, 0.45, 0.85, 0.7};
};

struct PrunedNode {
  NodeId node = kInvalidNode;
  std::uint32_t distance = 0;
  double relevance = 0.0;
  double centrality = 0.0;
  double score = 0.0;
  std::uint32_t token_cost = 0;
  bool seed = false;
};

struct PruneResult {
  std::vector<PrunedNode> nodes;  // selected nodes, sorted by score (desc)
  std::vector<Edge> edges;        // edges induced by the selected nodes
  std::uint64_t tokens_used = 0;
  std::uint64_t token_budget = 0;
  std::uint64_t candidates = 0;   // k-hop frontier size before pruning
};

struct GraphStats {
  std::size_t nodes = 0;
  std::size_t edges = 0;
  std::size_t memory_bytes = 0;
  std::array<std::size_t, kEdgeKindCount> edges_by_kind{};
  std::uint32_t max_out_degree = 0;
  std::uint32_t max_in_degree = 0;
};

class GraphEngine {
 public:
  GraphEngine() = default;
  GraphEngine(const GraphEngine&) = delete;
  GraphEngine& operator=(const GraphEngine&) = delete;

  // ---- construction -----------------------------------------------------
  NodeId add_node(std::string_view name, std::string_view kind = "symbol",
                  std::string_view file = "", std::int32_t line = 0,
                  std::uint32_t token_cost = 1);
  void add_edge(NodeId src, NodeId dst, EdgeKind kind = EdgeKind::Call);
  bool add_edge_by_name(std::string_view src, std::string_view dst,
                        EdgeKind kind = EdgeKind::Call);
  void add_edges(std::span<const NodeId> src, std::span<const NodeId> dst,
                 EdgeKind kind = EdgeKind::Call);
  void add_anonymous_nodes(std::size_t count);  // fast path for synthetic graphs
  void reserve(std::size_t nodes, std::size_t edges);
  void set_token_cost(NodeId id, std::uint32_t cost);
  void finalize();
  void clear();

  // ---- inspection -------------------------------------------------------
  [[nodiscard]] std::size_t num_nodes() const;
  [[nodiscard]] std::size_t num_edges();
  [[nodiscard]] std::optional<NodeId> find(std::string_view name) const;
  [[nodiscard]] NodeInfo node(NodeId id) const;
  [[nodiscard]] std::vector<NodeId> neighbors(NodeId id, Direction dir);
  [[nodiscard]] std::vector<Edge> edges();
  [[nodiscard]] std::vector<std::uint32_t> degrees(Direction dir);
  [[nodiscard]] GraphStats stats();

  // ---- algorithms -------------------------------------------------------
  // Breadth-first expansion up to k hops from every seed at once.
  [[nodiscard]] std::vector<HopResult> k_hop(std::span<const NodeId> seeds,
                                             std::uint32_t k, Direction dir);
  // Shortest path via bidirectional BFS (meet in the middle).
  [[nodiscard]] std::vector<NodeId> shortest_path(NodeId src, NodeId dst,
                                                  Direction dir = Direction::Out);
  // Global PageRank (cached until the graph changes).
  [[nodiscard]] std::vector<double> pagerank(double damping = 0.85,
                                             std::uint32_t max_iter = 100,
                                             double tol = 1e-10);
  // PageRank with the teleport vector concentrated on seed nodes.
  [[nodiscard]] std::vector<double> personalized_pagerank(
      std::span<const NodeId> seeds, std::span<const double> weights,
      double damping = 0.85, std::uint32_t max_iter = 100, double tol = 1e-10);
  // Token-budget subgraph pruner used by the GraphRAG retriever.
  [[nodiscard]] PruneResult prune(std::span<const NodeId> seeds,
                                  std::span<const double> seed_scores,
                                  const PruneOptions& opts);

 private:
  struct StringHash {
    using is_transparent = void;
    std::size_t operator()(std::string_view s) const noexcept {
      return std::hash<std::string_view>{}(s);
    }
  };

  void finalize_locked();
  void ensure_scratch_locked();
  std::uint32_t next_stamp_locked();
  std::vector<double> pagerank_locked(const std::vector<double>* teleport,
                                      double damping, std::uint32_t max_iter,
                                      double tol);
  const std::vector<double>& cached_pagerank_locked(double damping,
                                                    std::uint32_t max_iter,
                                                    double tol);
  [[nodiscard]] std::size_t memory_bytes_locked() const;
  template <typename Fn>
  void for_each_neighbor(NodeId u, Direction dir, Fn&& fn) const;

  mutable std::mutex mu_;

  std::vector<NodeInfo> nodes_;
  std::unordered_map<std::string, NodeId, StringHash, std::equal_to<>> index_;

  std::vector<Edge> edges_;  // canonical edge list (deduplicated on finalize)
  bool dirty_ = false;

  // CSR adjacency.
  std::vector<std::uint32_t> out_off_, in_off_;
  std::vector<NodeId> out_adj_, in_adj_;
  std::vector<EdgeKind> out_kind_, in_kind_;

  // Traversal scratch.
  std::vector<std::uint32_t> stamp_a_, stamp_b_, stamp_c_;
  std::vector<std::uint32_t> dist_a_, dist_b_;
  std::vector<NodeId> parent_a_, parent_b_;
  std::vector<double> rel_;
  std::uint32_t stamp_ = 0;

  // PageRank cache.
  std::vector<double> pr_cache_;
  double pr_damping_ = -1.0;
  std::uint32_t pr_iter_ = 0;
  double pr_tol_ = -1.0;
};

}  // namespace chronos
