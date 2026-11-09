#include "graph_engine.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <string>

namespace chronos {

namespace {

constexpr std::size_t kind_index(EdgeKind k) noexcept {
  return static_cast<std::size_t>(k);
}

constexpr Direction reverse(Direction d) noexcept {
  switch (d) {
    case Direction::Out: return Direction::In;
    case Direction::In: return Direction::Out;
    default: return Direction::Both;
  }
}

template <typename T>
std::size_t vec_bytes(const std::vector<T>& v) noexcept {
  return v.capacity() * sizeof(T);
}

}  // namespace

// ---------------------------------------------------------------------------
// construction
// ---------------------------------------------------------------------------

NodeId GraphEngine::add_node(std::string_view name, std::string_view kind,
                             std::string_view file, std::int32_t line,
                             std::uint32_t token_cost) {
  std::scoped_lock lock(mu_);
  if (auto it = index_.find(name); it != index_.end()) {
    NodeInfo& info = nodes_[it->second];
    if (!kind.empty() && kind != "symbol") info.kind = std::string(kind);
    if (!file.empty()) info.file = std::string(file);
    if (line != 0) info.line = line;
    if (token_cost > 1) info.token_cost = token_cost;
    return it->second;
  }
  if (nodes_.size() >= static_cast<std::size_t>(kInvalidNode)) {
    throw std::length_error("chronos: node limit reached");
  }
  const auto id = static_cast<NodeId>(nodes_.size());
  nodes_.push_back(NodeInfo{std::string(name), std::string(kind), std::string(file),
                            line, std::max<std::uint32_t>(token_cost, 1)});
  index_.emplace(std::string(name), id);
  dirty_ = true;
  return id;
}

void GraphEngine::add_anonymous_nodes(std::size_t count) {
  std::scoped_lock lock(mu_);
  if (nodes_.size() + count >= static_cast<std::size_t>(kInvalidNode)) {
    throw std::length_error("chronos: node limit reached");
  }
  nodes_.resize(nodes_.size() + count, NodeInfo{"", "symbol", "", 0, 1});
  dirty_ = true;
}

void GraphEngine::add_edge(NodeId src, NodeId dst, EdgeKind kind) {
  std::scoped_lock lock(mu_);
  if (src >= nodes_.size() || dst >= nodes_.size()) {
    throw std::out_of_range("chronos: edge endpoint out of range");
  }
  edges_.push_back(Edge{src, dst, kind});
  dirty_ = true;
}

bool GraphEngine::add_edge_by_name(std::string_view src, std::string_view dst,
                                   EdgeKind kind) {
  std::scoped_lock lock(mu_);
  const auto a = index_.find(src);
  const auto b = index_.find(dst);
  if (a == index_.end() || b == index_.end()) return false;
  edges_.push_back(Edge{a->second, b->second, kind});
  dirty_ = true;
  return true;
}

void GraphEngine::add_edges(std::span<const NodeId> src, std::span<const NodeId> dst,
                            EdgeKind kind) {
  if (src.size() != dst.size()) {
    throw std::invalid_argument("chronos: src and dst must have equal length");
  }
  std::scoped_lock lock(mu_);
  const auto n = nodes_.size();
  edges_.reserve(edges_.size() + src.size());
  for (std::size_t i = 0; i < src.size(); ++i) {
    if (src[i] >= n || dst[i] >= n) {
      throw std::out_of_range("chronos: edge endpoint out of range");
    }
    edges_.push_back(Edge{src[i], dst[i], kind});
  }
  dirty_ = true;
}

void GraphEngine::reserve(std::size_t nodes, std::size_t edges) {
  std::scoped_lock lock(mu_);
  nodes_.reserve(nodes);
  index_.reserve(nodes);
  edges_.reserve(edges);
}

void GraphEngine::set_token_cost(NodeId id, std::uint32_t cost) {
  std::scoped_lock lock(mu_);
  if (id >= nodes_.size()) throw std::out_of_range("chronos: node out of range");
  nodes_[id].token_cost = std::max<std::uint32_t>(cost, 1);
}

void GraphEngine::finalize() {
  std::scoped_lock lock(mu_);
  finalize_locked();
}

void GraphEngine::clear() {
  std::scoped_lock lock(mu_);
  nodes_.clear();
  index_.clear();
  edges_.clear();
  out_off_.clear();
  in_off_.clear();
  out_adj_.clear();
  in_adj_.clear();
  out_kind_.clear();
  in_kind_.clear();
  stamp_a_.clear();
  stamp_b_.clear();
  stamp_c_.clear();
  dist_a_.clear();
  dist_b_.clear();
  parent_a_.clear();
  parent_b_.clear();
  rel_.clear();
  pr_cache_.clear();
  pr_damping_ = -1.0;
  stamp_ = 0;
  dirty_ = false;
}

void GraphEngine::finalize_locked() {
  if (!dirty_ && out_off_.size() == nodes_.size() + 1) return;
  const std::size_t n = nodes_.size();

  // Deduplicate (src, dst) pairs; keep the lowest EdgeKind for each pair.
  std::ranges::sort(edges_, [](const Edge& a, const Edge& b) {
    if (a.src != b.src) return a.src < b.src;
    if (a.dst != b.dst) return a.dst < b.dst;
    return a.kind < b.kind;
  });
  const auto tail = std::ranges::unique(
      edges_, [](const Edge& a, const Edge& b) { return a.src == b.src && a.dst == b.dst; });
  edges_.erase(tail.begin(), tail.end());

  const std::size_t m = edges_.size();
  if (m >= std::numeric_limits<std::uint32_t>::max()) {
    throw std::length_error("chronos: edge limit reached");
  }

  out_off_.assign(n + 1, 0);
  in_off_.assign(n + 1, 0);
  for (const Edge& e : edges_) {
    ++out_off_[e.src + 1];
    ++in_off_[e.dst + 1];
  }
  std::inclusive_scan(out_off_.begin(), out_off_.end(), out_off_.begin());
  std::inclusive_scan(in_off_.begin(), in_off_.end(), in_off_.begin());

  out_adj_.resize(m);
  out_kind_.resize(m);
  in_adj_.resize(m);
  in_kind_.resize(m);
  std::vector<std::uint32_t> cursor(in_off_.begin(), in_off_.end() - 1);
  for (std::size_t i = 0; i < m; ++i) {
    const Edge& e = edges_[i];
    out_adj_[i] = e.dst;  // edges_ is sorted by src, so this matches out_off_
    out_kind_[i] = e.kind;
    const auto pos = cursor[e.dst]++;
    in_adj_[pos] = e.src;
    in_kind_[pos] = e.kind;
  }

  ensure_scratch_locked();
  pr_cache_.clear();
  pr_damping_ = -1.0;
  dirty_ = false;
}

void GraphEngine::ensure_scratch_locked() {
  const std::size_t n = nodes_.size();
  if (stamp_a_.size() == n) return;
  stamp_a_.resize(n, 0);
  stamp_b_.resize(n, 0);
  stamp_c_.resize(n, 0);
  dist_a_.resize(n, 0);
  dist_b_.resize(n, 0);
  parent_a_.resize(n, kInvalidNode);
  parent_b_.resize(n, kInvalidNode);
  rel_.resize(n, 0.0);
}

std::uint32_t GraphEngine::next_stamp_locked() {
  if (++stamp_ == 0) {  // wrapped: reset every stamp array once
    std::ranges::fill(stamp_a_, 0u);
    std::ranges::fill(stamp_b_, 0u);
    std::ranges::fill(stamp_c_, 0u);
    stamp_ = 1;
  }
  return stamp_;
}

template <typename Fn>
void GraphEngine::for_each_neighbor(NodeId u, Direction dir, Fn&& fn) const {
  if (dir != Direction::In) {
    for (auto i = out_off_[u], end = out_off_[u + 1]; i < end; ++i) fn(out_adj_[i], out_kind_[i]);
  }
  if (dir != Direction::Out) {
    for (auto i = in_off_[u], end = in_off_[u + 1]; i < end; ++i) fn(in_adj_[i], in_kind_[i]);
  }
}

// ---------------------------------------------------------------------------
// inspection
// ---------------------------------------------------------------------------

std::size_t GraphEngine::num_nodes() const {
  std::scoped_lock lock(mu_);
  return nodes_.size();
}

std::size_t GraphEngine::num_edges() {
  std::scoped_lock lock(mu_);
  finalize_locked();
  return edges_.size();
}

std::optional<NodeId> GraphEngine::find(std::string_view name) const {
  std::scoped_lock lock(mu_);
  if (auto it = index_.find(name); it != index_.end()) return it->second;
  return std::nullopt;
}

NodeInfo GraphEngine::node(NodeId id) const {
  std::scoped_lock lock(mu_);
  if (id >= nodes_.size()) throw std::out_of_range("chronos: node out of range");
  return nodes_[id];
}

std::vector<NodeId> GraphEngine::neighbors(NodeId id, Direction dir) {
  std::scoped_lock lock(mu_);
  finalize_locked();
  if (id >= nodes_.size()) throw std::out_of_range("chronos: node out of range");
  std::vector<NodeId> out;
  for_each_neighbor(id, dir, [&](NodeId v, EdgeKind) { out.push_back(v); });
  std::ranges::sort(out);
  const auto tail = std::ranges::unique(out);
  out.erase(tail.begin(), tail.end());
  return out;
}

std::vector<Edge> GraphEngine::edges() {
  std::scoped_lock lock(mu_);
  finalize_locked();
  return edges_;
}

std::vector<std::uint32_t> GraphEngine::degrees(Direction dir) {
  std::scoped_lock lock(mu_);
  finalize_locked();
  const std::size_t n = nodes_.size();
  std::vector<std::uint32_t> deg(n, 0);
  for (std::size_t u = 0; u < n; ++u) {
    if (dir != Direction::In) deg[u] += out_off_[u + 1] - out_off_[u];
    if (dir != Direction::Out) deg[u] += in_off_[u + 1] - in_off_[u];
  }
  return deg;
}

std::size_t GraphEngine::memory_bytes_locked() const {
  std::size_t bytes = vec_bytes(nodes_) + vec_bytes(edges_) + vec_bytes(out_off_) +
                      vec_bytes(in_off_) + vec_bytes(out_adj_) + vec_bytes(in_adj_) +
                      vec_bytes(out_kind_) + vec_bytes(in_kind_) + vec_bytes(stamp_a_) +
                      vec_bytes(stamp_b_) + vec_bytes(stamp_c_) + vec_bytes(dist_a_) +
                      vec_bytes(dist_b_) + vec_bytes(parent_a_) + vec_bytes(parent_b_) +
                      vec_bytes(rel_) + vec_bytes(pr_cache_);
  for (const NodeInfo& info : nodes_) {
    bytes += info.name.capacity() + info.kind.capacity() + info.file.capacity();
  }
  bytes += index_.size() * (sizeof(std::string) + sizeof(NodeId) + 2 * sizeof(void*));
  return bytes;
}

GraphStats GraphEngine::stats() {
  std::scoped_lock lock(mu_);
  finalize_locked();
  GraphStats s;
  s.nodes = nodes_.size();
  s.edges = edges_.size();
  for (const Edge& e : edges_) ++s.edges_by_kind[kind_index(e.kind)];
  for (std::size_t u = 0; u < s.nodes; ++u) {
    s.max_out_degree = std::max(s.max_out_degree, out_off_[u + 1] - out_off_[u]);
    s.max_in_degree = std::max(s.max_in_degree, in_off_[u + 1] - in_off_[u]);
  }
  s.memory_bytes = memory_bytes_locked();
  return s;
}

// ---------------------------------------------------------------------------
// traversal
// ---------------------------------------------------------------------------

std::vector<HopResult> GraphEngine::k_hop(std::span<const NodeId> seeds, std::uint32_t k,
                                          Direction dir) {
  std::scoped_lock lock(mu_);
  finalize_locked();
  const std::size_t n = nodes_.size();
  const std::uint32_t s = next_stamp_locked();

  std::vector<HopResult> result;
  std::vector<NodeId> frontier;
  std::vector<NodeId> next;
  frontier.reserve(seeds.size());
  for (const NodeId seed : seeds) {
    if (seed >= n) throw std::out_of_range("chronos: seed out of range");
    if (stamp_a_[seed] == s) continue;
    stamp_a_[seed] = s;
    result.push_back({seed, 0});
    frontier.push_back(seed);
  }
  for (std::uint32_t depth = 1; depth <= k && !frontier.empty(); ++depth) {
    next.clear();
    for (const NodeId u : frontier) {
      for_each_neighbor(u, dir, [&](NodeId v, EdgeKind) {
        if (stamp_a_[v] == s) return;
        stamp_a_[v] = s;
        result.push_back({v, depth});
        next.push_back(v);
      });
    }
    frontier.swap(next);
  }
  return result;
}

std::vector<NodeId> GraphEngine::shortest_path(NodeId src, NodeId dst, Direction dir) {
  std::scoped_lock lock(mu_);
  finalize_locked();
  const std::size_t n = nodes_.size();
  if (src >= n || dst >= n) throw std::out_of_range("chronos: node out of range");
  if (src == dst) return {src};

  const std::uint32_t s = next_stamp_locked();
  stamp_a_[src] = s;
  dist_a_[src] = 0;
  parent_a_[src] = kInvalidNode;
  stamp_b_[dst] = s;
  dist_b_[dst] = 0;
  parent_b_[dst] = kInvalidNode;

  std::vector<NodeId> fa{src};
  std::vector<NodeId> fb{dst};
  std::vector<NodeId> next;
  NodeId meet = kInvalidNode;
  std::uint32_t best = std::numeric_limits<std::uint32_t>::max();

  while (!fa.empty() && !fb.empty() && meet == kInvalidNode) {
    // Expand whichever side has the smaller frontier.
    const bool forward = fa.size() <= fb.size();
    auto& frontier = forward ? fa : fb;
    auto& mine = forward ? stamp_a_ : stamp_b_;
    auto& theirs = forward ? stamp_b_ : stamp_a_;
    auto& my_dist = forward ? dist_a_ : dist_b_;
    auto& their_dist = forward ? dist_b_ : dist_a_;
    auto& my_parent = forward ? parent_a_ : parent_b_;
    const Direction d = forward ? dir : reverse(dir);

    next.clear();
    for (const NodeId u : frontier) {
      for_each_neighbor(u, d, [&](NodeId v, EdgeKind) {
        if (mine[v] == s) return;
        mine[v] = s;
        my_dist[v] = my_dist[u] + 1;
        my_parent[v] = u;
        next.push_back(v);
        if (theirs[v] == s) {
          const std::uint32_t total = my_dist[v] + their_dist[v];
          if (total < best) {
            best = total;
            meet = v;
          }
        }
      });
    }
    frontier.swap(next);
  }
  if (meet == kInvalidNode) return {};

  std::vector<NodeId> path;
  for (NodeId x = meet; x != kInvalidNode; x = parent_a_[x]) path.push_back(x);
  std::ranges::reverse(path);
  for (NodeId x = parent_b_[meet]; x != kInvalidNode; x = parent_b_[x]) path.push_back(x);
  return path;
}

// ---------------------------------------------------------------------------
// PageRank
// ---------------------------------------------------------------------------

std::vector<double> GraphEngine::pagerank_locked(const std::vector<double>* teleport,
                                                 double damping, std::uint32_t max_iter,
                                                 double tol) {
  const std::size_t n = nodes_.size();
  if (n == 0) return {};
  const double inv_n = 1.0 / static_cast<double>(n);
  std::vector<double> rank(n, inv_n);
  std::vector<double> next(n, 0.0);
  std::vector<double> contrib(n, 0.0);

  for (std::uint32_t iter = 0; iter < max_iter; ++iter) {
    double dangling = 0.0;
    for (std::size_t u = 0; u < n; ++u) {
      const auto deg = out_off_[u + 1] - out_off_[u];
      if (deg == 0) {
        dangling += rank[u];
        contrib[u] = 0.0;
      } else {
        contrib[u] = rank[u] / static_cast<double>(deg);
      }
    }
    // Teleport and dangling mass are both redistributed along the teleport
    // vector, which matches NetworkX semantics.
    const double spread = damping * dangling + (1.0 - damping);
    double err = 0.0;
    for (std::size_t v = 0; v < n; ++v) {
      double sum = 0.0;
      for (auto i = in_off_[v], end = in_off_[v + 1]; i < end; ++i) sum += contrib[in_adj_[i]];
      const double tp = teleport ? (*teleport)[v] : inv_n;
      next[v] = damping * sum + spread * tp;
      err += std::fabs(next[v] - rank[v]);
    }
    rank.swap(next);
    if (err < tol) break;
  }
  return rank;
}

const std::vector<double>& GraphEngine::cached_pagerank_locked(double damping,
                                                               std::uint32_t max_iter,
                                                               double tol) {
  finalize_locked();
  if (pr_cache_.size() != nodes_.size() || pr_damping_ != damping || pr_iter_ != max_iter ||
      pr_tol_ != tol) {
    pr_cache_ = pagerank_locked(nullptr, damping, max_iter, tol);
    pr_damping_ = damping;
    pr_iter_ = max_iter;
    pr_tol_ = tol;
  }
  return pr_cache_;
}

std::vector<double> GraphEngine::pagerank(double damping, std::uint32_t max_iter, double tol) {
  if (damping < 0.0 || damping >= 1.0) throw std::invalid_argument("chronos: damping in [0, 1)");
  std::scoped_lock lock(mu_);
  return cached_pagerank_locked(damping, max_iter, tol);
}

std::vector<double> GraphEngine::personalized_pagerank(std::span<const NodeId> seeds,
                                                       std::span<const double> weights,
                                                       double damping, std::uint32_t max_iter,
                                                       double tol) {
  if (damping < 0.0 || damping >= 1.0) throw std::invalid_argument("chronos: damping in [0, 1)");
  if (!weights.empty() && weights.size() != seeds.size()) {
    throw std::invalid_argument("chronos: weights must match seeds");
  }
  std::scoped_lock lock(mu_);
  finalize_locked();
  const std::size_t n = nodes_.size();
  std::vector<double> teleport(n, 0.0);
  double total = 0.0;
  for (std::size_t i = 0; i < seeds.size(); ++i) {
    if (seeds[i] >= n) throw std::out_of_range("chronos: seed out of range");
    const double w = weights.empty() ? 1.0 : std::max(0.0, weights[i]);
    teleport[seeds[i]] += w;
    total += w;
  }
  if (total <= 0.0) return cached_pagerank_locked(damping, max_iter, tol);
  for (double& t : teleport) t /= total;
  return pagerank_locked(&teleport, damping, max_iter, tol);
}

// ---------------------------------------------------------------------------
// Token-budget subgraph pruner
// ---------------------------------------------------------------------------
//
// 1. Relevance propagation: seeds start with their (normalised) retrieval
//    score; every hop multiplies by hop_decay * kind_weight[edge kind]. This is
//    a max-product relaxation bounded to max_hops edges (Jacobi rounds, so a
//    path never exceeds the hop limit).
// 2. Each candidate is scored as a blend of propagated relevance and global
//    PageRank centrality.
// 3. A greedy knapsack fills the token budget: strong seeds first, then every
//    other candidate by score / (cost / mean_cost) ^ cost_exponent.
PruneResult GraphEngine::prune(std::span<const NodeId> seeds,
                               std::span<const double> seed_scores,
                               const PruneOptions& opts) {
  if (!seed_scores.empty() && seed_scores.size() != seeds.size()) {
    throw std::invalid_argument("chronos: seed_scores must match seeds");
  }
  std::scoped_lock lock(mu_);
  finalize_locked();
  PruneResult res;
  res.token_budget = opts.token_budget;
  const std::size_t n = nodes_.size();
  if (n == 0 || seeds.empty()) return res;

  const std::vector<double>& pr = cached_pagerank_locked(0.85, 100, 1e-10);
  const std::uint32_t s = next_stamp_locked();

  double max_seed = 0.0;
  for (std::size_t i = 0; i < seeds.size(); ++i) {
    max_seed = std::max(max_seed, seed_scores.empty() ? 1.0 : seed_scores[i]);
  }
  if (max_seed <= 0.0) max_seed = 1.0;

  // stamp_a_: touched, stamp_b_: is-seed, stamp_c_ + dist_b_: queued this round.
  std::vector<NodeId> touched;
  for (std::size_t i = 0; i < seeds.size(); ++i) {
    const NodeId v = seeds[i];
    if (v >= n) throw std::out_of_range("chronos: seed out of range");
    const double sc = std::clamp((seed_scores.empty() ? 1.0 : seed_scores[i]) / max_seed, 0.0, 1.0);
    if (stamp_a_[v] != s) {
      stamp_a_[v] = s;
      stamp_b_[v] = s;
      rel_[v] = sc;
      dist_a_[v] = 0;
      touched.push_back(v);
    } else {
      rel_[v] = std::max(rel_[v], sc);
    }
  }

  std::vector<std::pair<NodeId, double>> current;
  current.reserve(touched.size());
  for (const NodeId v : touched) current.emplace_back(v, rel_[v]);
  std::vector<NodeId> queued;

  for (std::uint32_t round = 1; round <= opts.max_hops && !current.empty(); ++round) {
    queued.clear();
    for (const auto& [u, ru] : current) {
      for_each_neighbor(u, opts.direction, [&](NodeId v, EdgeKind kind) {
        const double cand = ru * opts.hop_decay * opts.kind_weight[kind_index(kind)];
        if (cand <= 0.0) return;
        if (stamp_a_[v] != s) {
          stamp_a_[v] = s;
          rel_[v] = 0.0;
          dist_a_[v] = round;
          touched.push_back(v);
        }
        if (cand > rel_[v] + 1e-15) {
          rel_[v] = cand;
          dist_a_[v] = round;
          if (!(stamp_c_[v] == s && dist_b_[v] == round)) {
            stamp_c_[v] = s;
            dist_b_[v] = round;
            queued.push_back(v);
          }
        }
      });
    }
    current.clear();
    for (const NodeId v : queued) current.emplace_back(v, rel_[v]);
  }
  res.candidates = touched.size();

  double max_pr = 0.0;
  double mean_cost = 0.0;
  for (const NodeId v : touched) {
    max_pr = std::max(max_pr, pr[v]);
    mean_cost += nodes_[v].token_cost;
  }
  mean_cost = std::max(1.0, mean_cost / static_cast<double>(touched.size()));
  if (max_pr <= 0.0) max_pr = 1.0;

  const double cw = std::clamp(opts.centrality_weight, 0.0, 1.0);
  std::vector<PrunedNode> cands;
  std::vector<double> priority;
  cands.reserve(touched.size());
  for (const NodeId v : touched) {
    PrunedNode c;
    c.node = v;
    c.distance = dist_a_[v];
    c.relevance = rel_[v];
    c.centrality = pr[v] / max_pr;
    c.score = (1.0 - cw) * c.relevance + cw * c.centrality;
    c.token_cost = nodes_[v].token_cost;
    c.seed = stamp_b_[v] == s;
    cands.push_back(c);
  }
  std::vector<std::size_t> order(cands.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  priority.resize(cands.size());
  for (std::size_t i = 0; i < cands.size(); ++i) {
    const double rel_cost = std::max(1.0, static_cast<double>(cands[i].token_cost)) / mean_cost;
    priority[i] = cands[i].score / std::pow(rel_cost, opts.cost_exponent);
  }
  // Strong seeds are guaranteed a slot; weaker seeds compete with their graph
  // neighbours on value per token like every other candidate.
  auto forced = [&](std::size_t i) { return cands[i].seed && cands[i].relevance >= opts.seed_floor; };
  std::ranges::sort(order, [&](std::size_t a, std::size_t b) {
    if (forced(a) != forced(b)) return forced(a);
    if (forced(a)) return cands[a].score > cands[b].score;
    if (priority[a] != priority[b]) return priority[a] > priority[b];
    return cands[a].node < cands[b].node;
  });

  const std::uint32_t sel = next_stamp_locked();
  for (const std::size_t i : order) {
    if (opts.max_nodes != 0 && res.nodes.size() >= opts.max_nodes) break;
    const auto cost = cands[i].token_cost;
    if (res.tokens_used + cost > opts.token_budget) continue;
    res.tokens_used += cost;
    res.nodes.push_back(cands[i]);
    stamp_c_[cands[i].node] = sel;
  }
  std::ranges::sort(res.nodes, [](const PrunedNode& a, const PrunedNode& b) {
    if (a.score != b.score) return a.score > b.score;
    return a.node < b.node;
  });

  for (const PrunedNode& pn : res.nodes) {
    const NodeId u = pn.node;
    for (auto i = out_off_[u], end = out_off_[u + 1]; i < end; ++i) {
      if (stamp_c_[out_adj_[i]] == sel) res.edges.push_back(Edge{u, out_adj_[i], out_kind_[i]});
    }
  }
  return res;
}

}  // namespace chronos
