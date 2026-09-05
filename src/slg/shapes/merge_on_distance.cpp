/***************************************************************************
 * Copyright 1998-2025 by authors (see AUTHORS.txt)                        *
 *                                                                         *
 *   This file is part of LuxCoreRender.                                   *
 *                                                                         *
 * Licensed under the Apache License, Version 2.0 (the "License");         *
 * you may not use this file except in compliance with the License.        *
 * You may obtain a copy of the License at                                 *
 *                                                                         *
 *     http://www.apache.org/licenses/LICENSE-2.0                          *
 *                                                                         *
 * Unless required by applicable law or agreed to in writing, software     *
 * distributed under the License is distributed on an "AS IS" BASIS,       *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.*
 * See the License for the specific language governing permissions and     *
 * limitations under the License.                                          *
 ***************************************************************************/

// This shape modifier will merge vertices whose interdistance is lower than
// a given threshold

#include <cassert>
#include <vector>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <tuple>
#include <format>
#include <execution>

#include "oneapi/tbb.h"
#include "oneapi/tbb/scalable_allocator.h"
#include "oneapi/tbb/cache_aligned_allocator.h"

#include "luxrays/core/exttrianglemesh.h"
#include "slg/shapes/merge_on_distance.h"
#include "slg/scene/scene.h"
#include "luxrays/utils/utils.h"

using luxrays::Point;
using luxrays::WallClockTime;
using namespace oneapi::tbb;


namespace {



class NearlyEqualComparator {
public:

	explicit NearlyEqualComparator(const unsigned int tolerance) :
		boundmin(minus_epsilon * float(tolerance)),
		boundmax(plus_epsilon * float(tolerance))
	{}
	inline bool compare(const float a, const float b) const {
		const float delta1 = b - a;
		const float delta2 = a - b;
		const bool res1 = (boundmin <= delta1 and delta1 <= boundmax);
		const bool res2 = (boundmin <= delta2 and delta2 <= boundmax);
		return res1 or res2;
	}

private:
	inline static const float minus_epsilon =
		std::nextafter(0.f, std::numeric_limits<float>::lowest());

	inline static const float plus_epsilon =
		std::nextafter(0.f, std::numeric_limits<float>::max());

	const float boundmin;
	const float boundmax;
};


inline bool compare_points(
	const luxrays::Point& p1,
	const luxrays::Point& p2,
	const NearlyEqualComparator& comparator
) {
	const bool m0 = comparator.compare(p1[0], p2[0]);
	const bool m1 = comparator.compare(p1[1], p2[1]);
	const bool m2 = comparator.compare(p1[2], p2[2]);
	return m0 && m1 && m2;
}


// This is the classical Union-Find algorithm,
// in a parallel implementation (tbb powered)
class alignas(std::hardware_destructive_interference_size) UnionFind {
public:
    UnionFind() {}
    explicit UnionFind(const size_t count) {
		reserve(count);
	}
	UnionFind(const UnionFind& other) :
		parent(other.parent, Allocator()),
		rank(other.rank, Allocator())
	{}
	UnionFind(UnionFind&& other) :
		parent(std::move(other.parent)),
		rank(std::move(other.rank))
	{}
	UnionFind& operator=(const UnionFind& other) {
		parent = other.parent;
		rank = other.rank;
		return (*this);
	}
	UnionFind& operator=(UnionFind&& other) {
		parent = std::move(other.parent);
		rank = std::move(other.rank);
		return (*this);
	}

    // Find the root of the set containing element i
    u_int find(const u_int i) {
        if (parent.find(i) == parent.end()) {
            parent[i] = i;
            rank[i] = 0;
        }
        if (parent[i] != i) {
            parent[i] = find(parent[i]); // Path compression
        }
        return parent[i];
    }

    // Union the sets containing elements i and j
    void unite(const u_int i, const u_int j) {
        const u_int rootI = find(i);
        const u_int rootJ = find(j);

        if (rootI != rootJ) {
            // Union by rank
            if (rank[rootI] > rank[rootJ]) {
                parent[rootJ] = rootI;
            } else if (rank[rootI] < rank[rootJ]) {
                parent[rootI] = rootJ;
            } else {
                parent[rootJ] = rootI;
                rank[rootI]++;
            }
        }
    }

	// Reserve space
	void reserve(const size_t count) {
		parent.reserve(count);
		rank.reserve(count);
	}

    // Overload the += operator to merge two UnionFind instances
    UnionFind operator+=(const UnionFind& other) {

        for (const auto& pair : other.parent) {
            unite(pair.first, pair.second);
        }

        return (*this);
    }

	size_t size() const {
		return parent.size();
	}

	// Find without compression
	u_int find_readonly(const u_int i) const {
		auto res = parent.find(i);
        if (res != parent.end()) {
			return res->second;
		} else {
			return i;
		}

	}

private:
	using Allocator = tbb::cache_aligned_allocator<std::pair<const u_int, u_int>>;
	using Hash = std::hash<u_int>;
	using Equal = std::equal_to<u_int>;
    std::unordered_map<u_int, u_int, Hash, Equal, Allocator> parent;
    std::unordered_map<u_int, u_int, Hash, Equal, Allocator> rank;

    friend std::ostream& operator<<(std::ostream& os, const UnionFind& uf);
};

std::ostream& operator<<(std::ostream& os, const UnionFind& uf) {
    os << "Parent: ";
    for (const auto& pair : uf.parent) {
        os << "(" << pair.first << ", " << pair.second << ") ";
    }
    os << "\nRank: ";
    for (const auto& pair : uf.rank) {
        os << "(" << pair.first << ", " << pair.second << ") ";
    }
    return os;
}

// Cell Id, for partition indexing
// This id can be handled in two ways:
// - 3-uple of signed int 16
// - an unsigned int 64 (with first 16 bits set to zero)
union CellId {
	CellId(const int16_t x, const int16_t y, const int16_t z) {
		i16[0] = 0;
		i16[1] = x;
		i16[2] = y;
		i16[3] = z;
	}
	CellId(const uint64_t id) : u64(id) {
		if (i16[0]) {
			throw std::runtime_error("Invalid CellId");
		}
	}
	CellId(const CellId& other) : u64(other.u64) {};

	CellId(const CellId& other, int16_t dx, int16_t dy, int16_t dz) :
		u64(other.u64) {
		i16[1] += dx;
		i16[2] += dy;
		i16[3] += dz;
	};

	inline int16_t x() const { return i16[1]; }
	inline int16_t y() const { return i16[2]; }
	inline int16_t z() const { return i16[3]; }
	inline uint64_t id() const { return u64; }

	inline bool operator==(const CellId& other) const {
		return u64 == other.u64;
	}
	inline bool out_of_bound(int16_t dx, int16_t dy, int16_t dz) const {
		bool res =
			   (x() == INT16_MIN && dx == -1)
			or (x() == INT16_MAX && dx == +1)
			or (y() == INT16_MIN && dy == -1)
			or (y() == INT16_MAX && dy == +1)
			or (z() == INT16_MIN && dz == -1)
			or (z() == INT16_MAX && dz == +1);
		return res;
	}
private:
	uint64_t u64;
	int16_t i16[4];
};

}

// Introduce hash into std namespace
namespace std {
template <>
struct hash<CellId>
{
	std::size_t operator()(const CellId& x) const noexcept
	{
		std::size_t h = std::hash<uint64_t>{}(x.id());
		return h;
    }
};
}  // namespace std

namespace {

// Grid for spatial partitioning
class Grid {
public:
	Grid(
		const Point& p_origin,
		float p_cellSizeX,
		float p_cellSizeY,
		float p_cellSizeZ
	) :
		m_origin(p_origin),
		m_cellSizeX(p_cellSizeX),
		m_cellSizeY(p_cellSizeY),
		m_cellSizeZ(p_cellSizeZ)
	{};

	const Point& origin() const { return m_origin; }
	const float cellSizeX() const { return m_cellSizeX; }
	const float cellSizeY() const { return m_cellSizeY; }
	const float cellSizeZ() const { return m_cellSizeZ; }

private:
	const Point m_origin;
	const float m_cellSizeX, m_cellSizeY, m_cellSizeZ;

};



// Create a grid, ie give an origin point (or midPoint) and cell sizes on X, Y,
// Z
Grid ComputeGrid(const Point * points, u_int numPoints) {

	constexpr float minlimit = std::numeric_limits<float>::min();
	constexpr float maxlimit = std::numeric_limits<float>::max();

	using sixfloats = std::tuple<float, float, float, float, float, float>;

	constexpr size_t grain = 1024;

	// Compute bounding box
	auto boundingbox = parallel_reduce(
		// Range
		blocked_range<const Point*>(points, points+numPoints, grain),

		// Init
		std::make_tuple(maxlimit, maxlimit, maxlimit, minlimit, minlimit, minlimit),

		// Body
		[](const blocked_range<const Point*>& r, sixfloats init) -> sixfloats {
			auto [minX, minY, minZ, maxX, maxY, maxZ] = init;
			for(auto p=r.begin(); p!=r.end(); ++p) {
				minX = std::min(minX, p->x);
				minY = std::min(minY, p->y);
				minZ = std::min(minZ, p->z);
				maxX = std::max(maxX, p->x);
				maxY = std::max(maxY, p->y);
				maxZ = std::max(maxZ, p->z);
			}
			return std::make_tuple(minX, minY, minZ, maxX, maxY, maxZ);
		},

		// Reduce
		[](const sixfloats& a, const sixfloats& b) {
			const auto [minXa, minYa, minZa, maxXa, maxYa, maxZa] = a;
			const auto [minXb, minYb, minZb, maxXb, maxYb, maxZb] = b;
			return std::make_tuple(
				std::min(minXa, minXb),
				std::min(minYa, minYb),
				std::min(minZa, minZb),
				std::max(maxXa, maxXb),
				std::max(maxYa, maxYb),
				std::max(maxZa, maxZb)
			);
		}
	);

	const auto [minX, minY, minZ, maxX, maxY, maxZ] = boundingbox;
	Point midPoint((minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2);

	// Compute cell sizes
	constexpr float numCells = static_cast<float>(1 << 16);
	float cellSizeX = (maxX - minX) / numCells;
	float cellSizeY = (maxY - minY) / numCells;
	float cellSizeZ = (maxZ - minZ) / numCells;

	// Make the cells slightly larger than the bounding box
	cellSizeX = std::nextafterf(cellSizeX, INFINITY);
	cellSizeY = std::nextafterf(cellSizeY, INFINITY);
	cellSizeZ = std::nextafterf(cellSizeZ, INFINITY);

	SDL_LOG("Merge On Distance - Grid dimensions: Origin = "
		<< midPoint << " Cell size = ("
		<< cellSizeX << ", " << cellSizeY << ", " << cellSizeZ << ")"
	)

	return Grid(midPoint, cellSizeX, cellSizeY, cellSizeZ);

}

// The partition object (and subobjects)

using PartitionPoint = std::pair<u_int, luxrays::Point>;

// Partition element: a point number and its coordinates, padded and aligned
struct alignas(std::hardware_destructive_interference_size)
PartitionElem : public PartitionPoint {
	PartitionElem(u_int id, const luxrays::Point point)
		: PartitionPoint(id, point) {}
};

using PartitionBucket = std::vector<
	PartitionElem,
	tbb::cache_aligned_allocator<PartitionElem>
>;
using PartitionAllocator = tbb::scalable_allocator<
	std::pair<const CellId, PartitionBucket>
>;
using PartitionHash = std::hash<CellId>;
using PartitionEqual = std::equal_to<CellId>;

using Partition = std::unordered_map<
	CellId,
	PartitionBucket,
	PartitionHash,
	PartitionEqual,
	PartitionAllocator
>;

// Assign points to grid (do the partitioning)
Partition AssignPointsToGrid(
	const Grid& grid, const Point * points, u_int numPoints
) {
	// Avoid tiny sets of data for body
	constexpr size_t grain = 1024;

	auto partition = tbb::parallel_reduce(
		// Range
		blocked_range<u_int>(0, numPoints, grain),

		// Init
		Partition(numPoints),

		// Body
		[&](const blocked_range<u_int>& r, Partition&& partition) -> Partition {
			for (u_int i = r.begin(); i != r.end(); ++i) {
				const auto point = points[i];

				const auto p = point - grid.origin();
				const auto cellX = static_cast<int16_t>(p.x / grid.cellSizeX());
				const auto cellY = static_cast<int16_t>(p.y / grid.cellSizeY());
				const auto cellZ = static_cast<int16_t>(p.z / grid.cellSizeZ());
				const CellId cellId(cellX, cellY, cellZ);

				partition[cellId].emplace_back(i, point);
			}
			return partition;
		},

		// Reduce
		[](Partition&& partition1, Partition&& partition2) -> Partition {
			partition1.merge(partition2);
			for (const auto& [cellId, partitionBucket2] : partition2) {
				auto& partitionBucket1 = partition1[cellId];
				partitionBucket1.reserve(partitionBucket1.size() + partitionBucket2.size());
				partitionBucket1.insert(
					partitionBucket1.end(), partitionBucket2.begin(), partitionBucket2.end()
				);
			}
			return partition1;
		}
	);

	// Debug
#if 0
	size_t sup = 0;
	size_t count = 0;
	for (auto const& [key, value] : partition) {
		sup = std::max(value.size(), sup);
		count += value.size();
	}
	SDL_LOG("Grid sup/total: " << sup << " " << count);
#endif

	return partition;
}

constexpr std::array<std::array<int, 3>, 27> adjacency() {
	std::array<std::array<int, 3>, 27> res;
	size_t i = 0;
	for (auto dx : {-1, 0, 1}) {
		for (auto dy : {-1, 0, 1}) {
			for (auto dz : {-1, 0, 1}) {
				res[i++] = {dx, dy, dz};
			}
		}
	}
	return res;
}



// Gather equivalent points ("equivalent" meaning located at zero distance from
// each others)
// Points have previously been assigned to grid cells, so that we
// just compare points within same cells (saves a lot of time)
// 'tolerance' is a parameter for float comparison
class PointGrouping {
	const Partition& partition;
	const NearlyEqualComparator& comparator;
	const u_int numPoints;
	static constexpr auto ADJACENCY = adjacency();

public:

	UnionFind dsu;

	// Constructor (plain)
	PointGrouping(
		const Partition& p_partition,
		const NearlyEqualComparator& p_comparator,
		u_int p_numPoints
	) :
		partition(p_partition),
		comparator(p_comparator),
		numPoints(p_numPoints),
		dsu(UnionFind(numPoints))
	{}

	// Constructor (split)
	PointGrouping(PointGrouping& x, tbb::split):
		partition(x.partition),
		comparator(x.comparator),
		numPoints(x.numPoints),
		dsu(UnionFind(numPoints))
	{}

	// Body
	void operator()(const tbb::blocked_range<size_t>& r) {
		auto it = partition.begin();
		std::advance(it, r.begin());
		for (auto i = r.begin(); i != r.end(); ++i, ++it) {
			const auto& [cellId, cellPoints] = *it;

			for (const auto& [idx, curPoint] : cellPoints) {
				// Check adjacent cells
				for (const auto [dx, dy, dz] : ADJACENCY) {
					// Skip out-of-bound cells
					if (cellId.out_of_bound(dx, dy, dz)) continue;
					const CellId adjCellId(cellId, dx, dy, dz);

					const auto adjIt = partition.find(adjCellId);
					if (adjIt == partition.end()) continue;
					const auto& adjPoints = adjIt->second;

					// For each point in current cell and for each point
					// in adjacent cell, compute distance
					for (const auto& [adjIdx, adjPoint] : adjPoints) {
						if (idx >= adjIdx) continue;
						if (compare_points(curPoint, adjPoint, comparator)) {
							dsu.unite(idx, adjIdx);
						}
					}
				}  // for dx, dy, dz
			}
		}  // for i
	}  // operator()

	// Reduction
	void join(PointGrouping& rhs) {
		if (dsu.size() < rhs.dsu.size()) {
			std::swap(dsu, rhs.dsu);
		}
		dsu += rhs.dsu;
	}

};


UnionFind GroupPoints(const Partition& partition, u_int numPoints, u_int tolerance) {

	static tbb::affinity_partitioner tbb_partitioner;

	// Debug
#if 0
	size_t sup = 0;
	size_t count = 0;
	for (auto const& [key, value] : partition) {
		sup = std::max(value.size(), sup);
		count += value.size();
	}
	SDL_LOG("Grid sup/total: " << sup << " " << count);
#endif

	// Avoid tiny sets of data for body
	//constexpr size_t grain = 1024;
	constexpr size_t grain = 1024;
	const auto partition_size = partition.size();
	const NearlyEqualComparator comparator(tolerance);

	PointGrouping ptg(partition, comparator, numPoints);
	tbb::parallel_reduce(
		tbb::blocked_range<size_t>(0, partition_size, grain),
		ptg,
		tbb_partitioner
	);
	return ptg.dsu;
}


using Cluster = std::vector<u_int, tbb::scalable_allocator<u_int>>;

using ClusterMap = std::unordered_map<
	u_int,
	Cluster,
	std::hash<u_int>,
	std::equal_to<u_int>,
	tbb::scalable_allocator<std::pair<const u_int, Cluster>>
>;

// Group equivalent points into clusters
ClusterMap CreateClusters(const UnionFind& dsu, u_int numPoints) {
	// Avoid tiny sets of data for body
	constexpr size_t grain = 1024;

	// Get clusters as a map of vectors
	const auto clusters = tbb::parallel_reduce(
		// Range
		tbb::blocked_range<u_int>(0, numPoints, grain),

		// Init
		ClusterMap(grain),

		// Body
		[&dsu](const tbb::blocked_range<u_int>& r, ClusterMap&& map)
			-> ClusterMap
		{
			for (auto i = r.begin(); i != r.end(); ++i) {
				const auto clusterIndex = dsu.find_readonly(i);
				map[clusterIndex].push_back(i);
			}
			return map;
		},

		// Reduce
		[](ClusterMap&& map1, ClusterMap&& map2)
			-> ClusterMap
		{
			if (map1.size() < map2.size()) {
				std::swap(map1, map2);
			}

			map1.reserve(map1.size() + map2.size());

			map1.merge(map2);

			for(const auto& p: map2) {
				auto& cluster1 = map1[p.first];
				auto& cluster2 = p.second;
				cluster1.reserve(cluster1.size() + cluster2.size());
				cluster1.insert(
					cluster1.end(),
					std::make_move_iterator(cluster2.begin()),
					std::make_move_iterator(cluster2.end())
				);
			}

			return map1;
		}
	);

	return clusters;
}


// Merge nearby points, with spatial partioning acceleration
//
// This is the entry point of the merging algorithm
//
// Spatial partioning means that the points are first partitioned into a grid,
// and the distance computations are made only between points within the same
// grid cell and adjacent cells. This saves a lot of computations in case of
// large collection of points
//
// Returns:
// - The merged points, in the form of clusters (map of vectors)
//
ClusterMap mergePoints(
	const Point * points, u_int numPoints, u_int tolerance, bool importNormals
) {

	// Compute grid for spatial partitioning
	const Grid grid{ComputeGrid(points, numPoints)};

	// Assign points to grids cells (in other words: partition)
	const Partition partition{
		AssignPointsToGrid(grid, points, numPoints)
	};

	// For each cell, compare points within the cell and adjacent cells
	// and gather points at (nearly) zero distance from each others.
	// Gathering is made via a Union Find algo
	const UnionFind dsu{GroupPoints(partition, numPoints, tolerance)};

	// Finally, reformat result into convenient cluster format (vector of
	// vectors)
	const ClusterMap clusters{CreateClusters(dsu, numPoints)};

	return clusters;

}


// Recreate a mesh, based on a source mesh and a clusterisation
// Replace each variable with interpolated value
luxrays::ExtTriangleMeshUPtr RecreateMesh(
	const luxrays::ExtTriangleMesh& srcMesh,
	const ClusterMap& clustermap,
	bool importNormals
) {
	const auto numPoints = srcMesh.GetTotalVertexCount();
	const auto srcPoints = srcMesh.GetVertices();
	const auto numNewPoints = clustermap.size();

	// Nota: we use smart pointers until the creation of the ExtTriangleMesh,
	// so that the memory is automatically released if something goes wrong in
	// the process. Please keep it so.

	std::vector<Cluster> clusters;
	clusters.reserve(numNewPoints);
	std::transform(
		std::make_move_iterator(clustermap.begin()),
		std::make_move_iterator(clustermap.end()),
		std::back_inserter(clusters),
		[](auto &kv){ return kv.second;}
	);

	std::vector<u_int> pointMap(numPoints);

	// Allocate mesh data structures
	// Points
	std::unique_ptr<Point> newPoints{
		luxrays::ExtTriangleMesh::AllocVerticesBuffer(numNewPoints)
	};
	auto newPointsPtr = newPoints.get();

	// Normals
	std::unique_ptr<luxrays::Normal> newNormals;
	const auto srcNormals = srcMesh.GetNormals();
	if (srcMesh.HasNormals()) {
		newNormals.reset(new luxrays::Normal[numNewPoints]);
	}
	auto newNormalsPtr = importNormals ? newNormals.get() : nullptr;

	// UV
	luxrays::ExtMeshProp<luxrays::UV> newUVs;
	luxrays::ExtMeshProp<luxrays::UV> srcUVs;
	for (u_int i = 0; i < EXTMESH_MAX_DATA_COUNT; ++i) {
		if (srcMesh.HasUVs(i)) {
			srcUVs.SetLayer(i, srcMesh.GetUVs(i), numPoints);
			newUVs.AllocateLayer(i, numNewPoints);
		}
	}

	// Colors
	luxrays::ExtMeshProp<luxrays::Spectrum> newColors, srcColors;
	for (u_int i = 0; i < EXTMESH_MAX_DATA_COUNT; ++i) {
		if (srcMesh.HasColors(i)) {
			srcColors[i] = srcMesh.GetColors(i);
			newColors.AllocateLayer(i, numNewPoints);
		}
	}

	// Alphas
	luxrays::ExtMeshProp<float> newAlphas, srcAlphas;
	for (u_int i = 0; i < EXTMESH_MAX_DATA_COUNT; ++i) {
		if (srcMesh.HasAlphas(i)) {
			srcAlphas[i] = srcMesh.GetAlphas(i);
			newAlphas.AllocateLayer(i, numNewPoints);
		}
	}

	// VertexAOVs
	luxrays::ExtMeshProp<float>  newVertexAOVs, srcVertexAOVs;
	for (u_int i = 0; i < EXTMESH_MAX_DATA_COUNT; ++i) {
		if (srcMesh.HasVertexAOV(i)) {
			srcVertexAOVs[i] = srcMesh.GetVertexAOVs(i);
			newVertexAOVs.AllocateLayer(i, numNewPoints);
		}
	}

	// Avoid tiny sets of data for tbb body
	constexpr size_t grain = 1024;

	// Compute merged values of points, normals, uv etc.
	tbb::parallel_for(
        tbb::blocked_range<size_t>(0, numNewPoints, grain),

		[&](tbb::blocked_range<size_t>& r) {

			for (auto newIdx = r.begin(); newIdx != r.end(); ++newIdx) {
				const auto& cluster = clusters[newIdx];
				auto cluster_size = cluster.size();
				if (!cluster_size) continue;

				// Compute point map (mapping between old and new points)
				for (auto oldIdx: cluster) {
					pointMap[oldIdx] = newIdx;
				}

				// Compute merged points (centroids)
				Point newPoint = std::transform_reduce(
					cluster.cbegin(),
					cluster.cend(),
					Point(0, 0, 0),
					std::plus{},
					[&srcPoints](auto idx) -> Point { return srcPoints[idx]; }
				) / cluster_size;
				newPointsPtr[newIdx] = newPoint;

				// Compute merged normals
				if (importNormals && srcMesh.HasNormals()) {
					luxrays::Normal newNormal = std::transform_reduce(
						cluster.cbegin(),
						cluster.cend(),
						luxrays::Normal(0, 0, 0),
						std::plus{},
						[&srcNormals](auto idx) -> luxrays::Normal {
							return srcNormals[idx];
						}
					) / cluster_size;
					const float newNormalLength = newNormal.Length();
					if (newNormalLength) {
						newNormal /= newNormalLength;
					}
					newNormalsPtr[newIdx] = newNormal;
				}

				// Compute merged uv
				for (u_int i = 0; i < EXTMESH_MAX_DATA_COUNT; ++i) {
					if (srcMesh.HasUVs(i)) {
						auto newUVPtr = newUVs[i].get();
						luxrays::UV newUV = std::transform_reduce(
							cluster.cbegin(),
							cluster.cend(),
							luxrays::UV(0, 0),
							std::plus{},
							[&srcUVs, &i](auto idx) -> luxrays::UV {
								return srcUVs[i][idx];
							}
						) / cluster_size;
						newUVs[i].get()[newIdx] = newUV;
					}
				}

				// Compute merged colors
				for (u_int i = 0; i < EXTMESH_MAX_DATA_COUNT; ++i) {
					if (srcMesh.HasColors(i)) {
						auto newColorsPtr = newColors[i].get();
						luxrays::Spectrum newColor = std::transform_reduce(
							cluster.cbegin(),
							cluster.cend(),
							luxrays::Spectrum(0, 0, 0),
							std::plus{},
							[&srcColors, &i](auto idx) -> luxrays::Spectrum {
								return srcColors[i][idx];
							}
						) / cluster_size;
						newColorsPtr[newIdx] = newColor;
					}
				}

				// Compute merged alphas
				for (u_int i = 0; i < EXTMESH_MAX_DATA_COUNT; ++i) {
					if (srcMesh.HasAlphas(i)) {
						auto newAlphasPtr = newAlphas[i].get();
						float newAlpha = std::transform_reduce(
							cluster.cbegin(),
							cluster.cend(),
							0.f,
							std::plus{},
							[&srcAlphas, &i](auto idx) -> float {
								return srcAlphas[i][idx];
							}
						) / cluster_size;
						newAlphasPtr[newIdx] = newAlpha;
					}
				}

				// Compute merged vertex AOVs
				for (u_int i = 0; i < EXTMESH_MAX_DATA_COUNT; ++i) {
					if (srcMesh.HasVertexAOV(i)) {
						auto newVertexAOVsPtr = newVertexAOVs[i].get();
						float newVertexAOV = std::transform_reduce(
							cluster.cbegin(),
							cluster.cend(),
							0.f,
							std::plus{},
							[&srcVertexAOVs, &i](auto idx) -> float {
								return srcVertexAOVs[i][idx];
							}
						) / cluster_size;
						newVertexAOVsPtr[newIdx] = newVertexAOV;
					}
				}
			}  // lambda main for loop
		}  // Body lambda
	);  // tbb::parallel_for

	// Recompute triangles
	const u_int numTriangles = srcMesh.GetTotalTriangleCount();
	SDL_LOG("Number of triangles " << numTriangles);
	auto oldTriangles = srcMesh.GetTriangles();
	auto newTriangles = std::unique_ptr<luxrays::Triangle>(
		luxrays::ExtTriangleMesh::AllocTrianglesBuffer(numTriangles)
	);
	auto newTrianglesPtr = newTriangles.get();
	tbb::parallel_for(
		tbb::blocked_range<u_int>(0, numTriangles),
		[&](const tbb::blocked_range<u_int>& r) {
			for (u_int i = r.begin(); i != r.end(); ++i) {
				const auto& oldTriangle = oldTriangles[i];
				auto newTriangle = luxrays::Triangle(
					pointMap[oldTriangle.v[0]],
					pointMap[oldTriangle.v[1]],
					pointMap[oldTriangle.v[2]]
				);
				newTrianglesPtr[i] = newTriangle;
			}
		}
	);

	///TODO
	//// Create layer arrays (release smart pointers...)
	//ExtMeshProp<UV> meshUVs;
	//ExtMeshProp<Spectrum> meshCols;
	//ExtMeshProp<float> meshAlphas;

	//for (u_int i = 0; i < EXTMESH_MAX_DATA_COUNT; ++i) {
		//if (srcMesh.HasUVs(i)) {
			//(*meshUVs)[i] = newUVs[i].release();
		//} else {
			//(*meshUVs)[i] = nullptr;
		//}

		//if (srcMesh.HasColors(i)) {
			//(*meshCols)[i] = newColors[i].release();
		//} else {
			//(*meshCols)[i] = nullptr;
		//}

		//if (srcMesh.HasAlphas(i)) {
			//(*meshAlphas)[i] = newAlphas[i].release();
		//} else {
			//(*meshAlphas)[i] = nullptr;
		//}
	//}

	// Create new mesh
	auto newMesh = std::make_unique<luxrays::ExtTriangleMesh>(
		numNewPoints,
		numTriangles,
		newPoints.release(),
		newTriangles.release(),
		importNormals ? newNormals.release() : nullptr,
		newUVs,
		newColors,
		newAlphas,
		srcMesh.GetBevelRadius()
	);

	// Copy AOV to new mesh
	for (u_int i = 0; i < EXTMESH_MAX_DATA_COUNT; ++i) {
		if (srcMesh.HasVertexAOV(i)) {
			newMesh->SetVertexAOV(i, newVertexAOVs.GetLayer(i), newVertexAOVs.GetLayerSize());
		}
	}

	return newMesh;
}



} // namespace


namespace slg {

MergeOnDistanceShape::MergeOnDistanceShape(
	luxrays::ExtTriangleMeshRef  srcMesh,
	u_int tolerance,
	bool importNormals
) {

	SDL_LOG("Merge On Distance - Applying to " << srcMesh.GetName());

	const double startTime = WallClockTime();

	mesh = std::move(ApplyMergeOnDistance(srcMesh, tolerance, importNormals));

	const double endTime = WallClockTime();
	SDL_LOG(
		std::format(
			"Merge On Distance - Merging time: {:.3f} secs", endTime - startTime
		)
	);
}

MergeOnDistanceShape::~MergeOnDistanceShape() {
}

luxrays::ExtTriangleMeshUPtr
MergeOnDistanceShape::ApplyMergeOnDistance(
	luxrays::ExtTriangleMeshRef srcMesh,
	u_int tolerance,
	bool importNormals
) {


	// Get merged points
	auto clusters = mergePoints(
		srcMesh.GetVertices(),
		srcMesh.GetTotalVertexCount(),
		tolerance,
		importNormals
	);

	auto dstMesh = RecreateMesh(srcMesh, clusters, importNormals);


	SDL_LOG(
		"Merge On Distance - Reduced from "
		<< srcMesh.GetTotalVertexCount()
		<< " to "
		<< dstMesh->GetTotalVertexCount()
		<< " vertices"
	);

	return dstMesh;
}

luxrays::ExtTriangleMeshUPtr
MergeOnDistanceShape::RefineImpl(SceneConstRef scene) {
	return std::move(mesh);
}

}  // namespace slg

// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
