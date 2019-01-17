// Copyright © 2014-2018 the Surge contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

import Accelerate

public enum MatrixAxies {
    case row
    case column
}

public struct Matrix<Scalar> where Scalar: FloatingPoint, Scalar: ExpressibleByFloatLiteral {
    public let rows: Int
    public let columns: Int
    public var grid: [Scalar]

    public init(rows: Int, columns: Int, repeatedValue: Scalar) {
        self.rows = rows
        self.columns = columns

        self.grid = [Scalar](repeating: repeatedValue, count: rows * columns)
    }

    // For Matrix([[...], ...])
    public init<T: Collection, U: Collection>(_ contents: T, rows: Int? = nil, columns: Int? = nil) where
      T.Element == U, U.Element == Scalar
    {
        self.init(
          rows: rows ?? contents.count,
          columns: columns ?? (contents.first == nil ? 0 : contents.first!.count),
          repeatedValue: 0.0
        )

        // Optimized [TODO github PR]
        //  - Measurement: printTime { Matrix<Float>((0..<1000).map { i in Array(repeating: 3, count: 1000) }) }
        // Old [0.080s]
        // for (i, row) in contents.enumerated() {
        //     precondition(row.count == self.columns, "All rows should have the same number of columns")
        //     grid.replaceSubrange(i*self.columns ..< (i + 1)*self.columns, with: row)
        // }
        // New [0.015s]
        self.grid = contents.flatMap { [self] (row: U) -> [Scalar] in // [Scalar] i/o U else the deprecated flatMap (now compactMap)
            precondition(row.count == self.columns, "All rows should have the same number of columns")
            return Array(row)
        }

    }

    // For Matrix(rows, columns, [[...], ...]), when rows or columns is 0 but the other isn't
    public init<T: Collection, U: Collection>(rows: Int, columns: Int, gridRows: T) where
      T.Element == U, U.Element == Scalar
    {
      self.init(gridRows, rows: rows, columns: columns)
    }

    public init(row: [Scalar]) {
        self.init(rows: 1, columns: row.count, grid: row)
    }

    public init(column: [Scalar]) {
        self.init(rows: column.count, columns: 1, grid: column)
    }

    // init(columns:) is below, specialized to Float/Double i/o Scalar b/c transpose() requires it
    public init(rows: [[Scalar]]) {
        self.init(rows)
    }

    public init(rows: Int, columns: Int, grid: [Scalar]) {
        precondition(grid.count == rows * columns)

        self.rows = rows
        self.columns = columns

        self.grid = grid
    }

    public subscript(row: Int, column: Int) -> Scalar {
        get {
            assert(indexIsValidForRow(row, column: column))
            return grid[(row * columns) + column]
        }

        set {
            assert(indexIsValidForRow(row, column: column))
            grid[(row * columns) + column] = newValue
        }
    }

    public subscript(row row: Int) -> [Scalar] {
        get {
            assert(row < rows)
            let startIndex = row * columns
            let endIndex = row * columns + columns
            return Array(grid[startIndex..<endIndex])
        }

        set {
            assert(row < rows)
            assert(newValue.count == columns)
            let startIndex = row * columns
            let endIndex = row * columns + columns
            grid.replaceSubrange(startIndex..<endIndex, with: newValue)
        }
    }

    public subscript(column column: Int) -> [Scalar] {
        get {
            var result = [Scalar](repeating: 0.0, count: rows)
            for i in 0..<rows {
                let index = i * columns + column
                result[i] = self.grid[index]
            }
            return result
        }

        set {
            assert(column < columns)
            assert(newValue.count == rows)
            for i in 0..<rows {
                let index = i * columns + column
                grid[index] = newValue[i]
            }
        }
    }

    // Like np X[rows, :]
    //  - subscript(columns:) is below, specialized to Float/Double i/o Scalar b/c transpose() requires it
    public subscript(rows rows: Range<Int>) -> Matrix<Scalar> {
      get {
        return Matrix(
          rows: rows.count,
          columns: columns,
          gridRows: rows.map { self[row: $0] }
        )
      }
    }

    private func indexIsValidForRow(_ row: Int, column: Int) -> Bool {
        return row >= 0 && row < rows && column >= 0 && column < columns
    }

    public var shape: (Int, Int) {
      get { return (rows, columns) }
    }

    public var isEmpty: Bool {
      get { return rows * columns == 0 }
    }

    // Like np.flatten()
    public func flatten() -> [Scalar] {
      return grid
    }

    public func vect<X>(_ f: ([Scalar]) -> [X]) -> Matrix<X> {
      var result = Matrix<X>(rows: rows, columns: columns, repeatedValue: 0.0)
      result.grid = f(grid)
      return result
    }

    public func mapRows<X>(_ f: (ArraySlice<Scalar>) -> [X]) -> Matrix<X> {
      return Matrix<X>(
        rows: rows,
        columns: columns,
        gridRows: self.map(f) // self.map is row-major
      )
    }

    public func flipVertically() -> Matrix<Scalar> {
      return Matrix<Scalar>(
        rows: rows,
        columns: columns,
        gridRows: Array(self.reversed()) // row-major
      )
    }

}

extension Matrix where Scalar == Float {

  // Like np X.T
  public var T: Matrix {
    get { return transpose(self) }
  }

  public init(columns: [[Scalar]]) {
    self = Matrix(rows: columns).T
  }

  // Like np X[:, columns]
  public subscript(columns columns: Range<Int>) -> Matrix {
    get { return self.T[rows: columns].T }
  }

  // Like np X[rows, columns]
  public subscript(rows rows: Range<Int>, columns columns: Range<Int>) -> Matrix {
    get { return self[rows: rows].T[rows: columns].T }
  }

}

extension Matrix where Scalar == Double {

  // Like np X.T
  public var T: Matrix {
    get { return transpose(self) }
  }

  public init(columns: [[Scalar]]) {
    self = Matrix(rows: columns).T
  }

  // Like np X[:, columns]
  public subscript(columns columns: Range<Int>) -> Matrix {
    get { return self.T[rows: columns].T }
  }

  // Like np X[rows, columns]
  public subscript(rows rows: Range<Int>, columns columns: Range<Int>) -> Matrix {
    get { return self[rows: rows].T[rows: columns].T }
  }

}

// MARK: - Printable

extension Matrix: CustomStringConvertible {
    public var description: String {
        var description = ""

        for i in 0..<rows {
            let contents = (0..<columns).map({ "\(self[i, $0])" }).joined(separator: "\t")

            switch (i, rows) {
            case (0, 1):
                description += "(\t\(contents)\t)"
            case (0, _):
                description += "⎛\t\(contents)\t⎞"
            case (rows - 1, _):
                description += "⎝\t\(contents)\t⎠"
            default:
                description += "⎜\t\(contents)\t⎥"
            }

            description += "\n"
        }

        return description
    }
}

// MARK: - SequenceType

extension Matrix: Sequence {
    public func makeIterator() -> AnyIterator<ArraySlice<Scalar>> {
        let endIndex = rows * columns
        var nextRowStartIndex = 0

        return AnyIterator {
            if nextRowStartIndex == endIndex {
                return nil
            }

            let currentRowStartIndex = nextRowStartIndex
            nextRowStartIndex += self.columns

            return self.grid[currentRowStartIndex..<nextRowStartIndex]
        }
    }
}

extension Matrix: Equatable {}
public func ==<T> (lhs: Matrix<T>, rhs: Matrix<T>) -> Bool {
    return lhs.rows == rhs.rows && lhs.columns == rhs.columns && lhs.grid == rhs.grid
}

// MARK: -

public func add(_ x: Matrix<Float>, _ y: Matrix<Float>) -> Matrix<Float> {
    precondition(x.rows == y.rows && x.columns == y.columns, "Matrix dimensions not compatible with addition")

    var results = y
    results.grid.withUnsafeMutableBufferPointer { pointer in
        cblas_saxpy(Int32(x.grid.count), 1.0, x.grid, 1, pointer.baseAddress!, 1)
    }

    return results
}

public func add(_ x: Matrix<Double>, _ y: Matrix<Double>) -> Matrix<Double> {
    precondition(x.rows == y.rows && x.columns == y.columns, "Matrix dimensions not compatible with addition")

    var results = y
    results.grid.withUnsafeMutableBufferPointer { pointer in
        cblas_daxpy(Int32(x.grid.count), 1.0, x.grid, 1, pointer.baseAddress!, 1)
    }

    return results
}

public func sub(_ x: Matrix<Float>, _ y: Matrix<Float>) -> Matrix<Float> {
    precondition(x.rows == y.rows && x.columns == y.columns, "Matrix dimensions not compatible with subtraction")

    var results = y
    results.grid.withUnsafeMutableBufferPointer { pointer in
        catlas_saxpby(Int32(x.grid.count), 1.0, x.grid, 1, -1, pointer.baseAddress!, 1)
    }

    return results
}

public func sub(_ x: Matrix<Double>, _ y: Matrix<Double>) -> Matrix<Double> {
    precondition(x.rows == y.rows && x.columns == y.columns, "Matrix dimensions not compatible with subtraction")

    var results = y
    results.grid.withUnsafeMutableBufferPointer { pointer in
        catlas_daxpby(Int32(x.grid.count), 1.0, x.grid, 1, -1, pointer.baseAddress!, 1)
    }

    return results
}

public func mul(_ alpha: Float, _ x: Matrix<Float>) -> Matrix<Float> {
    var results = x
    results.grid.withUnsafeMutableBufferPointer { pointer in
        cblas_sscal(Int32(x.grid.count), alpha, pointer.baseAddress!, 1)
    }

    return results
}

public func mul(_ alpha: Double, _ x: Matrix<Double>) -> Matrix<Double> {
    var results = x
    results.grid.withUnsafeMutableBufferPointer { pointer in
        cblas_dscal(Int32(x.grid.count), alpha, pointer.baseAddress!, 1)
    }

    return results
}

public func mul(_ x: Matrix<Float>, _ y: Matrix<Float>) -> Matrix<Float> {
    precondition(x.columns == y.rows, "Matrix dimensions not compatible with multiplication")

    var results = Matrix<Float>(rows: x.rows, columns: y.columns, repeatedValue: 0.0)
    if results.rows > 0 && results.columns > 0 { // Avoid https://github.com/mattt/Surge/issues/92
      if x.columns > 0 { // HACK Avoid crash, mimic numpy (nonempty zero matrix) [TODO github issue]
        results.grid.withUnsafeMutableBufferPointer { pointer in
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(x.rows), Int32(y.columns), Int32(x.columns), 1.0, x.grid, Int32(x.columns), y.grid, Int32(y.columns), 0.0, pointer.baseAddress!, Int32(y.columns))
        }
      }
    }

    return results
}

public func mul(_ x: Matrix<Double>, _ y: Matrix<Double>) -> Matrix<Double> {
    precondition(x.columns == y.rows, "Matrix dimensions not compatible with multiplication")

    var results = Matrix<Double>(rows: x.rows, columns: y.columns, repeatedValue: 0.0)
    if results.rows > 0 && results.columns > 0 { // Avoid https://github.com/mattt/Surge/issues/92
      if x.columns > 0 { // HACK Avoid crash, mimic numpy (nonempty zero matrix) [TODO github issue]
        results.grid.withUnsafeMutableBufferPointer { pointer in
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(x.rows), Int32(y.columns), Int32(x.columns), 1.0, x.grid, Int32(x.columns), y.grid, Int32(y.columns), 0.0, pointer.baseAddress!, Int32(y.columns))
        }
      }
    }

    return results
}

public func elmul(_ x: Matrix<Double>, _ y: Matrix<Double>) -> Matrix<Double> {
    precondition(x.rows == y.rows && x.columns == y.columns, "Matrix must have the same dimensions")
    var result = Matrix<Double>(rows: x.rows, columns: x.columns, repeatedValue: 0.0)
    result.grid = x.grid .* y.grid
    return result
}

public func elmul(_ x: Matrix<Float>, _ y: Matrix<Float>) -> Matrix<Float> {
    precondition(x.rows == y.rows && x.columns == y.columns, "Matrix must have the same dimensions")
    var result = Matrix<Float>(rows: x.rows, columns: x.columns, repeatedValue: 0.0)
    result.grid = x.grid .* y.grid
    return result
}

public func div(_ x: Matrix<Double>, _ y: Matrix<Double>) -> Matrix<Double> {
    let yInv = inv(y)
    precondition(x.columns == yInv.rows, "Matrix dimensions not compatible")
    return mul(x, yInv)
}

public func div(_ x: Matrix<Float>, _ y: Matrix<Float>) -> Matrix<Float> {
    let yInv = inv(y)
    precondition(x.columns == yInv.rows, "Matrix dimensions not compatible")
    return mul(x, yInv)
}

public func pow(_ x: Matrix<Double>, _ y: Double) -> Matrix<Double> {
    var result = Matrix<Double>(rows: x.rows, columns: x.columns, repeatedValue: 0.0)
    result.grid = pow(x.grid, y)
    return result
}

public func pow(_ x: Matrix<Float>, _ y: Float) -> Matrix<Float> {
    var result = Matrix<Float>(rows: x.rows, columns: x.columns, repeatedValue: 0.0)
    result.grid = pow(x.grid, y)
    return result
}

public func exp(_ x: Matrix<Double>) -> Matrix<Double> {
    var result = Matrix<Double>(rows: x.rows, columns: x.columns, repeatedValue: 0.0)
    result.grid = exp(x.grid)
    return result
}

public func exp(_ x: Matrix<Float>) -> Matrix<Float> {
    var result = Matrix<Float>(rows: x.rows, columns: x.columns, repeatedValue: 0.0)
    result.grid = exp(x.grid)
    return result
}

public func sum(_ x: Matrix<Double>, axies: MatrixAxies = .column) -> Matrix<Double> {

    switch axies {
    case .column:
        var result = Matrix<Double>(rows: 1, columns: x.columns, repeatedValue: 0.0)
        for i in 0..<x.columns {
            result.grid[i] = sum(x[column: i])
        }
        return result

    case .row:
        var result = Matrix<Double>(rows: x.rows, columns: 1, repeatedValue: 0.0)
        for i in 0..<x.rows {
            result.grid[i] = sum(x[row: i])
        }
        return result
    }
}

public func sum(_ x: Matrix<Float>, axies: MatrixAxies = .column) -> Matrix<Float> {

    switch axies {
    case .column:
        var result = Matrix<Float>(rows: 1, columns: x.columns, repeatedValue: 0.0)
        for i in 0..<x.columns {
            result.grid[i] = sum(x[column: i])
        }
        return result

    case .row:
        var result = Matrix<Float>(rows: x.rows, columns: 1, repeatedValue: 0.0)
        for i in 0..<x.rows {
            result.grid[i] = sum(x[row: i])
        }
        return result
    }
}

public func inv(_ x: Matrix<Float>) -> Matrix<Float> {
    precondition(x.rows == x.columns, "Matrix must be square")

    // Mimic numpy i/o crashing [TODO github issue]
    if x.rows == 0 && x.columns == 0 {
      return x
    }

    var results = x

    var ipiv = [__CLPK_integer](repeating: 0, count: x.rows * x.rows)
    var lwork = __CLPK_integer(x.columns * x.columns)
    var work = [CFloat](repeating: 0.0, count: Int(lwork))
    var error: __CLPK_integer = 0
    var nc = __CLPK_integer(x.columns)

    withUnsafeMutablePointers(&nc, &lwork, &error) { nc, lwork, error in
        withUnsafeMutableMemory(&ipiv, &work, &(results.grid)) { ipiv, work, grid in
            sgetrf_(nc, nc, grid.pointer, nc, ipiv.pointer, error)
            sgetri_(nc, grid.pointer, nc, ipiv.pointer, work.pointer, lwork, error)
        }
    }

    assert(error == 0, "Matrix not invertible")

    return results
}

public func inv(_ x: Matrix<Double>) -> Matrix<Double> {
    precondition(x.rows == x.columns, "Matrix must be square")

    // Mimic numpy i/o crashing [TODO github issue]
    if x.rows == 0 && x.columns == 0 {
      return x
    }

    var results = x

    var ipiv = [__CLPK_integer](repeating: 0, count: x.rows * x.rows)
    var lwork = __CLPK_integer(x.columns * x.columns)
    var work = [CDouble](repeating: 0.0, count: Int(lwork))
    var error: __CLPK_integer = 0
    var nc = __CLPK_integer(x.columns)

    withUnsafeMutablePointers(&nc, &lwork, &error) { nc, lwork, error in
        withUnsafeMutableMemory(&ipiv, &work, &(results.grid)) { ipiv, work, grid in
            dgetrf_(nc, nc, grid.pointer, nc, ipiv.pointer, error)
            dgetri_(nc, grid.pointer, nc, ipiv.pointer, work.pointer, lwork, error)
        }
    }

    assert(error == 0, "Matrix not invertible")

    return results
}

public func transpose(_ x: Matrix<Float>) -> Matrix<Float> {
    var results = Matrix<Float>(rows: x.columns, columns: x.rows, repeatedValue: 0.0)
    results.grid.withUnsafeMutableBufferPointer { pointer in
        vDSP_mtrans(x.grid, 1, pointer.baseAddress!, 1, vDSP_Length(x.columns), vDSP_Length(x.rows))
    }

    return results
}

public func transpose(_ x: Matrix<Double>) -> Matrix<Double> {
    var results = Matrix<Double>(rows: x.columns, columns: x.rows, repeatedValue: 0.0)
    results.grid.withUnsafeMutableBufferPointer { pointer in
        vDSP_mtransD(x.grid, 1, pointer.baseAddress!, 1, vDSP_Length(x.columns), vDSP_Length(x.rows))
    }

    return results
}

/// Computes the matrix determinant.
public func det(_ x: Matrix<Float>) -> Float? {
    var decomposed = x
    var pivots = [__CLPK_integer](repeating: 0, count: min(x.rows, x.columns))
    var info = __CLPK_integer()
    var m = __CLPK_integer(x.rows)
    var n = __CLPK_integer(x.columns)
    _ = withUnsafeMutableMemory(&pivots, &(decomposed.grid)) { ipiv, grid in
        withUnsafeMutablePointers(&m, &n, &info) { m, n, info in
            sgetrf_(m, n, grid.pointer, m, ipiv.pointer, info)
        }
    }

    if info != 0 {
        return nil
    }

    var det = 1 as Float
    for (i, p) in zip(pivots.indices, pivots) {
        if p != i + 1 {
            det = -det * decomposed[i, i]
        } else {
            det = det * decomposed[i, i]
        }
    }
    return det
}

/// Computes the matrix determinant.
public func det(_ x: Matrix<Double>) -> Double? {
    var decomposed = x
    var pivots = [__CLPK_integer](repeating: 0, count: min(x.rows, x.columns))
    var info = __CLPK_integer()
    var m = __CLPK_integer(x.rows)
    var n = __CLPK_integer(x.columns)
    _ = withUnsafeMutableMemory(&pivots, &(decomposed.grid)) { ipiv, grid in
        withUnsafeMutablePointers(&m, &n, &info) { m, n, info in
            dgetrf_(m, n, grid.pointer, m, ipiv.pointer, info)
        }
    }

    if info != 0 {
        return nil
    }

    var det = 1 as Double
    for (i, p) in zip(pivots.indices, pivots) {
        if p != i + 1 {
            det = -det * decomposed[i, i]
        } else {
            det = det * decomposed[i, i]
        }
    }
    return det
}

// MARK: - Operators

public func + (lhs: Matrix<Float>, rhs: Matrix<Float>) -> Matrix<Float> {
    return add(lhs, rhs)
}

public func + (lhs: Matrix<Double>, rhs: Matrix<Double>) -> Matrix<Double> {
    return add(lhs, rhs)
}

public func - (lhs: Matrix<Float>, rhs: Matrix<Float>) -> Matrix<Float> {
    return sub(lhs, rhs)
}

public func - (lhs: Matrix<Double>, rhs: Matrix<Double>) -> Matrix<Double> {
    return sub(lhs, rhs)
}

public func + (lhs: Matrix<Float>, rhs: Float) -> Matrix<Float> {
    return Matrix(rows: lhs.rows, columns: lhs.columns, grid: lhs.grid + rhs)
}

public func + (lhs: Matrix<Double>, rhs: Double) -> Matrix<Double> {
    return Matrix(rows: lhs.rows, columns: lhs.columns, grid: lhs.grid + rhs)
}

public func * (lhs: Float, rhs: Matrix<Float>) -> Matrix<Float> {
    return mul(lhs, rhs)
}

public func * (lhs: Double, rhs: Matrix<Double>) -> Matrix<Double> {
    return mul(lhs, rhs)
}

public func * (lhs: Matrix<Float>, rhs: Matrix<Float>) -> Matrix<Float> {
    return mul(lhs, rhs)
}

public func * (lhs: Matrix<Double>, rhs: Matrix<Double>) -> Matrix<Double> {
    return mul(lhs, rhs)
}

public func / (lhs: Matrix<Double>, rhs: Matrix<Double>) -> Matrix<Double> {
    return div(lhs, rhs)
}

public func / (lhs: Matrix<Float>, rhs: Matrix<Float>) -> Matrix<Float> {
    return div(lhs, rhs)
}

public func / (lhs: Matrix<Double>, rhs: Double) -> Matrix<Double> {
    var result = Matrix<Double>(rows: lhs.rows, columns: lhs.columns, repeatedValue: 0.0)
    result.grid = lhs.grid / rhs
    return result
}

public func / (lhs: Matrix<Float>, rhs: Float) -> Matrix<Float> {
    var result = Matrix<Float>(rows: lhs.rows, columns: lhs.columns, repeatedValue: 0.0)
    result.grid = lhs.grid / rhs
    return result
}

postfix operator ′
public postfix func ′ (value: Matrix<Float>) -> Matrix<Float> {
    return transpose(value)
}

public postfix func ′ (value: Matrix<Double>) -> Matrix<Double> {
    return transpose(value)
}
