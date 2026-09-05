/***************************************************************************
 * Copyright 1998-2026 by authors (see AUTHORS.txt)                        *
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

/// Observer pointer
///
/// "a (not very) smart pointer type that takes no ownership responsibility for
/// its pointees, i.e., for the objects it observes. As such, it is intended as
/// a near drop-in replacement for raw pointer types, with the advantage that,
/// as a vocabulary type, it indicates its intended use without need for
/// detailed analysis by code readers."
///
/// Observer pointer is a non-owning, non-iterator, nullable pointer-like
/// object.
///
/// It has been unfairly criticized and reduced to a simple template<typename
/// T> using T*, but its capabilities are much greater:
///
/// - A default constructed observer_ptr will always be initialized to nullptr;
///   a regular pointer may or may not be initialized, depending on the context.
/// - observer_ptr only supports operations which make sense for a reference;
///   this enforces correct usage:
///     - operator[] is not implemented for observer_ptr, as this is an array
///       operation.
///     - Pointer arithmetic is not possible with observer_ptr, as these are
///       iterator operations.
/// - Two observer_ptrs have strict weak ordering on all implementations, which
///   is not guaranteed for two arbitrary pointers. This is because operator< is
///   implemented in terms of std::less for observer_ptr (as with std::unique_ptr
///   and std::shared_ptr).
/// - observer_ptr<void> appears to be unsupported, which may encourage the use
///   of safer solutions (e.g. std::any and std::variant)



#include <cstddef>	// for nullptr_t
#include <utility>	// for std::swap
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/nvp.hpp>

namespace luxrays {

template <typename T>
class observer_ptr {
    static_assert(
        !std::is_void_v<T>,
        "Template parameter T cannot be void"
    );
public:
	using element_type = std::remove_extent_t<T>;

	// Default constructor: initialize to nullptr
	constexpr observer_ptr() noexcept : ptr(nullptr) {}
	constexpr observer_ptr(std::nullptr_t) noexcept : ptr(nullptr) {}

	// Constructor from raw pointer (implicit conversions allowed)
	constexpr explicit observer_ptr(T* p) noexcept : ptr(p) {}

	// Copy constructor and assignment
	constexpr observer_ptr(const observer_ptr&) noexcept = default;
	constexpr observer_ptr& operator=(const observer_ptr&) noexcept = default;

	// Move constructor and assignment
	constexpr observer_ptr(observer_ptr&&) noexcept = default;
	constexpr observer_ptr& operator=(observer_ptr&&) noexcept = default;

	// Assignment from raw pointer
	constexpr observer_ptr& operator=(T* p) noexcept {
		ptr = p;
		return *this;
	}

        // Conversion to pointer
        constexpr operator T*() { return ptr; }
        constexpr operator const T*() const { return ptr; }

	// Destructor
	~observer_ptr() = default;

	// Dereference operators
	constexpr T& operator*() const noexcept { return *ptr; }
	constexpr T* operator->() const noexcept { return ptr; }

	// Conversion to bool (for boolean context)
	explicit operator bool() const noexcept { return ptr != nullptr; }

	// Get the raw pointer
	constexpr T* get() const noexcept { return ptr; }

	// Reset to nullptr
	constexpr void reset() noexcept { ptr = nullptr; }

	// Swap with another observer_ptr
	void swap(observer_ptr& other) noexcept { std::swap(ptr, other.ptr); }

	// Comparison operators
	friend constexpr bool operator==(const observer_ptr& lhs, const observer_ptr& rhs) noexcept {
		return lhs.ptr == rhs.ptr;
	}
	friend constexpr bool operator!=(const observer_ptr& lhs, const observer_ptr& rhs) noexcept {
		return lhs.ptr != rhs.ptr;
	}
	friend constexpr bool operator==(const observer_ptr& lhs, std::nullptr_t) noexcept {
		return lhs.ptr == nullptr;
	}
	friend constexpr bool operator!=(const observer_ptr& lhs, std::nullptr_t) noexcept {
		return lhs.ptr != nullptr;
	}

	// Comparison with raw pointers (const T*)
	friend constexpr bool operator==(const observer_ptr& lhs, const T* rhs) noexcept {
		return lhs.ptr == rhs;
	}
	friend constexpr bool operator!=(const observer_ptr& lhs, const T* rhs) noexcept {
		return lhs.ptr != rhs;
	}
	friend constexpr bool operator==(const T* lhs, const observer_ptr& rhs) noexcept {
		return lhs == rhs.ptr;
	}
	friend constexpr bool operator!=(const T* lhs, const observer_ptr& rhs) noexcept {
		return lhs != rhs.ptr;
	}

        // Converting constructor from observer_ptr<U> to observer_ptr<T>
        template <typename U, typename = std::enable_if_t<std::is_convertible_v<U*, T*>>>
        constexpr observer_ptr(const observer_ptr<U>& other) noexcept : ptr(other.get()) {}

        // Converting assignment operator from observer_ptr<U> to observer_ptr<T>
        template <typename U, typename = std::enable_if_t<std::is_convertible_v<U*, T*>>>
        constexpr observer_ptr& operator=(const observer_ptr<U>& other) noexcept {
            ptr = other.get();
            return *this;
        }


	// Boost.Serialization support
	template <typename Archive>
	void serialize(Archive& ar, const unsigned int /* version */) {
		ar & boost::serialization::make_nvp("ptr", ptr);
	}

private:
	T* ptr;
};


// Free function for dynamic casting observer_ptr<B> to observer_ptr<A>
template<class T, class U>
inline observer_ptr<T>
dynamic_observer_cast(const observer_ptr<U>& r) noexcept
{
	auto* p = dynamic_cast<typename observer_ptr<T>::element_type*>(r.get());

	if (p) return observer_ptr<T>(p);

	return observer_ptr<T>{nullptr};
}

template<class T, class U>
inline observer_ptr<const T>
dynamic_observer_cast(const observer_ptr<const U>& r) noexcept
{
	const auto* p = dynamic_cast<const typename observer_ptr<const T>::element_type*>(r.get());

	if (p) return observer_ptr<const T>(p);

	return observer_ptr<const T>{nullptr};
	}

template<class T, class U>
inline observer_ptr<T>
static_observer_cast(const observer_ptr<U>& r) noexcept
{
	auto* p = static_cast<typename observer_ptr<T>::element_type*>(r.get());

	if (p) return observer_ptr<T>(p);

	return observer_ptr<T>{nullptr};
}

template<class T, class U>
inline observer_ptr<const T>
static_observer_cast(const observer_ptr<const U>& r) noexcept
{
	const auto* p = static_cast<const typename observer_ptr<const T>::element_type*>(r.get());

	if (p) return observer_ptr<const T>(p);

	return observer_ptr<const T>{nullptr};
}

// Non-member swap
template <typename T>
void swap(observer_ptr<T>& lhs, observer_ptr<T>& rhs) noexcept {
	lhs.swap(rhs);
}


// Comparison
//
template<typename T, typename U>
bool
operator<(observer_ptr<T> p1, observer_ptr<U> p2)
{
  using myless_t = std::less<
      typename std::common_type< typename std::add_pointer<T>::type, typename std::add_pointer<U>::type >::type
  >;
  return myless_t{}(p1.get(), p2.get());
}

template<typename T, typename U>
bool
operator>(observer_ptr<T> p1, observer_ptr<U> p2)
{
  return p2 < p1;
}

template<typename T, typename U>
bool
operator<=(observer_ptr<T> p1, observer_ptr<U> p2)
{
  return !(p2 < p1);
}

template<typename T, typename U>
bool
operator>=(observer_ptr<T> p1, observer_ptr<U> p2)
{
  return !(p1 < p2);
}

} // namespace luxrays

// Boost.Serialization split free function for non-intrusive serialization
namespace boost {
namespace serialization {

template <typename Archive, typename T>
void save(Archive& ar, const luxrays::observer_ptr<T>& ptr, const unsigned int /* version */) {
	T* raw_ptr = ptr.get();
	ar << boost::serialization::make_nvp("ptr", raw_ptr);
}

template <typename Archive, typename T>
void load(Archive& ar, luxrays::observer_ptr<T>& ptr, const unsigned int /* version */) {
	T* raw_ptr;
	ar >> boost::serialization::make_nvp("ptr", raw_ptr);
	ptr = raw_ptr;
}

template <typename Archive, typename T>
void serialize(Archive& ar, luxrays::observer_ptr<T>& ptr, const unsigned int version) {
	boost::serialization::split_free(ar, ptr, version);
}

} // namespace serialization
} // namespace boost

// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
