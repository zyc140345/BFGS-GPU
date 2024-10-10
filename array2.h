#pragma once

#include <stdexcept>
#include <cstdlib>
#include <memory>
#include <cassert>
#include <cstring>

template <typename T>
class array2 {
public:
	array2() = default;

	array2(size_t rows, size_t cols, T default_value = T()) : rows_(rows), cols_(cols) {
		if ((data_ = (T*)malloc(rows_ * cols_ * sizeof(T))) == nullptr) {
			throw std::runtime_error("Cannot allocate memory");
		}

		std::uninitialized_fill_n(data_, rows * cols, default_value);
	}

	array2(const array2<T>& rhs) : rows_(rhs.rows_), cols_(rhs.cols_) {
		size_t bytes = rows_ * cols_ * sizeof(T);
		if ((data_ = (T*)malloc(bytes)) == nullptr) {
			throw std::runtime_error("Cannot allocate memory");
		}

		memcpy(data_, rhs.data_, bytes);
	}

	array2(array2<T>&& rhs) noexcept : data_{ rhs.data_ }, rows_{ rhs.rows_ }, cols_{ rhs.cols_ } {
		rhs.data_ = nullptr;
		rhs.rows_ = rhs.cols_ = 0;
	}

	~array2() {
		free(data_);
	}

	array2& operator=(const array2& rhs) {
		if (this != &rhs) {
			size_t bytes = rhs.rows_ * rhs.cols_ * sizeof(T);
			if (size() < rhs.size()) {
				T* new_data = (T*)realloc(data_, bytes);
				if (new_data == nullptr) {
					throw std::runtime_error("Cannot reallocate memory");
				}
				else if (data_ != new_data) {
					free(data_);
					data_ = new_data;
				}
			}

			memcpy(data_, rhs.data_, bytes);
		}

		return *this;
	}

	array2& operator=(array2&& rhs) noexcept {
		if (this != &rhs) {
			free(data_);
			data_ = rhs.data_;
			rows_ = rhs.rows_;
			cols_ = rhs.cols_;

			rhs.data_ = nullptr;
			rhs.rows_ = rhs.cols_ = 0;
		}

		return *this;
	}

	bool operator==(const array2& rhs) noexcept {
		if (rows_ != rhs.rows_ || cols_ != rhs.cols_) {
			return false;
		}

		return memcmp(data_, rhs.data_, rows_ * cols_ * sizeof(T)) == 0;
	}

	void resize(size_t rows, size_t cols, T val = T())
	{
		T* new_data = (T*)realloc(data_, rows * cols * sizeof(T));
		if (new_data == nullptr) {
			throw std::runtime_error("Cannot reallocate memory");
		}
		else if (new_data != data_) {
			free(data_);
			data_ = new_data;
		}

		rows_ = rows;
		cols_ = cols;
		std::uninitialized_fill_n(data_, rows_ * cols_, val);
	}

	void clear() {
		free(data_);
		data_ = nullptr;
		rows_ = cols_ = 0;
	}

	T operator()(size_t row, size_t col) const noexcept {
		assert(row < rows_ && col < cols_);
		return data_[row * cols_ + col];
	}

	T& operator()(size_t row, size_t col) noexcept {
		assert(row < rows_ && col < cols_);
		return data_[row * cols_ + col];
	}

	T operator[](size_t idx) const noexcept {
		assert(idx < rows_ * cols_);
		return data_[idx];
	}

	T& operator[](size_t idx) noexcept {
		assert(idx < rows_ * cols_);
		return data_[idx];
	}

	size_t rows() const noexcept { return rows_; }

	size_t cols() const noexcept { return cols_; }

	size_t size() const noexcept { return rows_ * cols_; }

	T* data() const noexcept { return data_; }

private:
	T* data_{ nullptr };
	size_t rows_{0};
	size_t cols_{0};
};

