#![allow(non_snake_case)]

//! Tools for fitting curves to sets of points.

/*! 
# General usage
The general interface to this module is through the `ParamCurveFit` trait, which is currently
implemented for `CubicBez` and `QuadBez`. These implementations allow for specifying one constraint
per each control point. Below is an example of fitting a freeform (unconstrained) cubic and
quadratic Bezier curves to 5 points.
```rust
use kurbo::{Point, CubicBez, QuadBez, fitting::{ParamCurveFit, Constraint::Free}};

let point_cloud: &[Point] = &[
    (0., 2.).into(),
    (1., 3.).into(),
    (2., 2.).into(),
    (1., 0.).into(),
    (3., 3.).into(),
];

let cubic_fit = CubicBez::fit(point_cloud, &[Free, Free, Free, Free]);
let quadratic_fit = QuadBez::fit(point_cloud, &[Free, Free, Free]);
```

# The algorithm
## The problem
The problem of fitting quadratic and cubic Bezier curves to sets of points does not have an exact
solution. If we knew which is the value of the parameter `t` closest to each point's
projection on the best-fit curve, the problem would reduce to a standard least squares problem,
solvable exactly using standard methods of linear algebra. However, to find the closest `t`, we
would need to already have the solution.

## The approach
This crate approaches this problem in a fashion similar to expectation-maximization: given an
initial guess for the curve C, we follow in a two-step iteration where each time we find the
closest `t` values given the current C and then solve the least squares fit using those `t`
values. In Python-like pseudocode, the algorithm looks as follows:

```python
def fit(points):
    proposal = initial_guess(points)

    while True:
        error = 0.
        approximate_ts = []

        for point in points:
            distance, t = project_on_curve(point, proposal)
            error += distance
            approximate_ts.append(t)

        if is_low_enough(error):
            return proposal

        proposal = solve_least_squares(points, approximate_ts)
```

This algorithm is not guaranteed to yield the best approximation to the input points, however
since both steps of the iteration reduce the error, this scheme is guaranteed to converge.
*/

use crate::{Point, Line, ParamCurveNearest};

use nalgebra::{
    U1, Dynamic, Matrix, VecStorage, Vector2 as GVector2
};

/// A parametrized curve which can be fitted to a set of points given constraints.
pub trait ParamCurveFit: ParamCurveNearest {
    type Constraints;

    /// Find the curve which best approximates a set of points in
    /// the least squares metric. Returns the mean fit error across all points
    /// and the fitted curve
    fn fit(points: &[Point], constaints: &Self::Constraints) -> (f64, Self);
}

/// Represents the three possible types of constraints imposed on control points
/// when fitting curves.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Constraint {

    /// This control point is unconstrained.
    Free,
    
    /// This control point is fixed to be exactly equal to `Point`. 
    /// A `From<Point>` implementation is provided
    /// to allow easy construction
    /// ```
    /// # use kurbo::{Point, fitting::Constraint};
    /// let p = Point::new(0., 0.);
    /// assert_eq!(Constraint::Fixed(p), p.into());
    /// ```
    Fixed(Point),
    
    /// This control point is constrained to lie on the line
    /// between the two points.
    /// A `From<(Point, Point)>` implementation is provided
    /// to allow easy construction
    /// ```
    /// # use kurbo::{Point, fitting::Constraint};
    /// let p1 = Point::new(0., 0.);
    /// let p2 = Point::new(1., 0.);
    /// assert_eq!(Constraint::Line(p1, p2), (p1, p2).into());
    /// ```
    Line(Point, Point),
}

impl Constraint {
    fn is_fixed(&self) -> bool {
        if let Constraint::Fixed(_) = *self {
            true
        } else {
            false
        }
    }

    fn get_fixed(&self) -> Option<Point> {
        if let Constraint::Fixed(p) = *self {
            Some(p)
        } else {
            None
        }
    }
}

impl From<Point> for Constraint {
    /// Constructs a `Constraint::Fixed` variant from the point
    /// ```
    /// # use kurbo::{Point, fitting::Constraint};
    /// let p = Point::new(0., 0.);
    /// assert_eq!(Constraint::Fixed(p), p.into());
    /// ```
    fn from(p: Point) -> Constraint {
        Constraint::Fixed(p)
    }
}

impl From<(Point, Point)> for Constraint {
    /// Constructs a `Constraint::Line` variant from the pair of points
    /// ```
    /// # use kurbo::{Point, fitting::Constraint};
    /// let p1 = Point::new(0., 0.);
    /// let p2 = Point::new(1., 0.);
    /// assert_eq!(Constraint::Line(p1, p2), (p1, p2).into());
    /// ```
    fn from(ps: (Point, Point)) -> Constraint {
        Constraint::Line(ps.0, ps.1)
    }
}

type VMatrix<Col, Row> = Matrix<f64, Col, Row, VecStorage<f64, Col, Row>>;

pub(crate) type DMatrix = VMatrix<Dynamic, Dynamic>;
pub(crate) type VectorN = VMatrix<Dynamic, U1>;
type Vector2 = GVector2<f64>;


// A helper trait to reduce code duplication in `fit_with_t` method below. Implemented
// on fittable types
pub(crate) trait FromPointIter {
    fn from_point_iter(iter: impl Iterator<Item = Point>) -> Self;
}

fn initial_guess<T: From<Line>>(points: &[Point]) -> T {
    let p0 = points.first().expect("failed to fetch the first point");
    let pn = points.last().expect("failed to fetch the last point");

    Line::new(*p0, *pn).into()
}

// builds a block-sparse matrix from the provided matrix such that
// given block B, we have that
//                [B  0]
// two_block(B) = [    ] 
//                [0  B]
fn two_block(block: &DMatrix) -> DMatrix {
    let shape = block.shape();

    DMatrix::from_fn(2 * shape.0, 2 * shape.1, |r, c| {
        let r_mod = r % shape.0;
        let r_div = r / shape.0;
        let c_mod = c % shape.1;
        let c_div = c / shape.1;

        if r_div == c_div {
            block[(r_mod, c_mod)]
        } else {
            0.
        }
    })
}


fn build_m(constraints: &[Constraint], fixed: bool, dof: usize, M: &DMatrix)
-> DMatrix
{
    assert_eq!(constraints.len(), dof);
    assert_eq!(M.shape().0, 2 * dof);
    assert_eq!(M.shape().1, 2 * dof);

    let mut columns = Vec::<usize>::with_capacity(2*dof);

    // pick X coordinate columns (0..DOF)
    for (i, constr) in constraints.iter().enumerate() {
        if fixed == constr.is_fixed() {
            columns.push(i);
        }
    }

    // pick Y coordinate columns (DOF..2*DOF)
    for (i, constr) in constraints.iter().enumerate() {
        if fixed == constr.is_fixed() {
            columns.push(i + dof);
        }
    }
    
    M.select_columns(columns.iter())
}

fn embed_add(a: &Point, b: &Point) -> (Vector2, Vector2) {
    let embed = Vector2::new(a.x - b.x, a.y - b.y);
    let add = Vector2::new(b.x, b.y);

    (embed, add)
}

fn build_embedding(constraints: &[Constraint]) -> (DMatrix, VectorN) {
    use Constraint::*;

    let mut n_dof = 0;
    let mut n_fit = 0;
    for constr in constraints {
        match constr {
            Fixed(_)   => continue,
            Line(_, _) => {
                n_dof += 1;
                n_fit += 2;
            },
            Free       => {
                n_dof += 2;
                n_fit += 2;
            },
        };
    };

    let mut embedding = DMatrix::zeros(n_fit, n_dof);
    let mut additive = VectorN::zeros(n_fit);

    let mut row = 0;
    let mut col = 0;
    debug_assert!(n_fit % 2 == 0);
    let skip = n_fit / 2;

    for constr in constraints {
        match constr {
            Fixed(_)   => continue,

            Line(a, b) => {
                let (embed, add) = embed_add(a, b);

                embedding[(row, col)] = embed[0];
                embedding[(row+skip, col)] = embed[1];
                additive[row] = add[0];
                additive[row+skip] = add[1]; 

                row += 1;
                col += 1;
            },

            Free       => {
                embedding[(row, col)] = 1.;
                embedding[(row+skip, col+1)] = 1.;

                row += 1;
                col += 2;
            },
        };
    };

    (embedding, additive)
}

fn build_mc_offset(constraints: &[Constraint], dof: usize, M: &DMatrix) -> VectorN {
    assert_eq!(constraints.len(), dof);
    assert_eq!(M.shape().0, 2 * dof);
    assert_eq!(M.shape().1, 2 * dof);

    let m = build_m(constraints, true, dof, M);
    let mut xs = vec![];
    let mut ys = vec![];

    for constr in constraints {
        if let Constraint::Fixed(point) = constr {
            xs.push(point.x);
            ys.push(point.y);
        }
    }

    assert_eq!(m.shape().1, xs.len() + ys.len());

    let c = VectorN::from_iterator(
        m.shape().1,

        xs.into_iter()
            .chain(ys.into_iter())
    );

    m * c
}

struct MatrixCache {
    dof: usize,
    M_free: DMatrix,
    P_base: VectorN,
    MC_offset: VectorN,
    embedding: DMatrix,
    additive: VectorN,
}

impl MatrixCache {
    fn new(points: &[Point], constraints: &[Constraint], dof: usize, M: &DMatrix) -> Self {
        assert_eq!(constraints.len(), dof);
        assert_eq!(M.shape().0, 2 * dof);
        assert_eq!(M.shape().1, 2 * dof);

        let M_free = build_m(constraints, false, dof, &M);

        let P_base = {
            let xs = points
                        .iter()
                        .map(|p| p.x);
            let ys = points
                        .iter()
                        .map(|p| p.y);

            VectorN::from_iterator(points.len() * 2, xs.chain(ys))
        };

        let MC_offset = build_mc_offset(constraints, dof, M);

        let (embedding, additive) = build_embedding(constraints);

        MatrixCache {
            dof: dof,
            M_free: M_free,
            P_base: P_base,
            MC_offset: MC_offset,
            embedding: embedding,
            additive: additive
        }
    }

    fn TM_and_P(&self, ts: &[f64]) -> (DMatrix, VectorN) {
        let T = {
            let T_small = DMatrix::from_fn(ts.len(), self.dof, |r, c| {
                ts[r].powi(c as i32)
            });

            two_block(&T_small)
        };

        let P_offset = &T * &self.MC_offset;

        let T_M_free = T * &self.M_free;
        let T_M_free_embedded = &T_M_free * &self.embedding;
        let additive_offset = &T_M_free * &self.additive;
        let P = &self.P_base - P_offset - additive_offset;

        (T_M_free_embedded, P)
    }

    fn print(&self) {
        println!("MatrixCache {{\n\tdof: {dof}\n\tM_free: {M_free}\n\tP_base: {P_base}\n\tMC_offset: {MC_offset}\n\tembedding: {embedding}\n\tadditive: {additive}\n}}", dof=self.dof,
        M_free=self.M_free, P_base=self.P_base, MC_offset=self.MC_offset, embedding=self.embedding, additive=self.additive);
    }
}

// TODO: improve performance by caching matrices such as `P_base`, `embedding`, `additive`,
// `P_offset` inside the `fit` parent instead of recomputing them each time inside `fit_with_t`
fn fit_with_t<T: FromPointIter>(
    points: &[Point], ts: &[f64], constraints: &[Constraint],
    dof: usize, M: &DMatrix, _matrix_cache: &MatrixCache) -> T
{
    assert_eq!(points.len(), ts.len());
    let matrix_cache = MatrixCache::new(points, constraints, dof, M);
    //matrix_cache.print();
    let (TM, P) = matrix_cache.TM_and_P(ts);

    // now we have T, M and P and can solve least squares for C
    let svd = TM.svd(true, true);
    let least_sqr = svd.solve(&P, 1e-6).expect("solve failed");

    // and reconstruct the bezier control points
    let mut ctrl_points = Vec::with_capacity(dof);
    let mut least_sqr_iter = least_sqr.iter();

    for constr in constraints {
        let ctrl_point = match constr {
            // for free constraints, the coordinates are given explicitly
            // in the least squares solution
            Constraint::Free => {
                let x = least_sqr_iter.next().unwrap();
                let y = least_sqr_iter.next().unwrap();
                Point::new(*x, *y)
            },

            // for line constraints, the solution contains only the parameter
            // `t` which specifies the point along the line at which our control
            // point should lie. We need to project it back to 2d
            Constraint::Line(a, b) => {
                let t = least_sqr_iter.next().unwrap();
                let (embed, add) = embed_add(a, b);
                let embedded = *t * embed + add;
                Point::new(embedded[0], embedded[1])
            },

            // for fixed constraints, we have no data in the least squares solution
            // but all we need is the constraint itself
            Constraint::Fixed(point) => *point,
        };

        ctrl_points.push(ctrl_point);
    };

    assert_eq!(ctrl_points.len(), dof);
    T::from_point_iter(ctrl_points.into_iter())
}

pub(crate) fn fit<T: FromPointIter + ParamCurveNearest + From<Line> + std::fmt::Debug>(
    points: &[Point], constraints: &[Constraint], dof: usize, M: &DMatrix)
-> (f64, T)
{

    const NEAREST_PREC: f64 = 1e-6; // TODO: how much?
    const STOP_TOL: f64 = 1.; // TODO: how much?
    const MAX_ITER: u32 = 64;

    let n_points = points.len();

    assert!(n_points > 0);

    // short circuit if the problem has no degrees of freedom
    if constraints.iter().all(Constraint::is_fixed) {
        let bezier = T::from_point_iter(
            constraints
                .iter()
                .map(|c| c.get_fixed().unwrap())
        );

        let total_error: f64 = points
                            .iter()
                            .map(|point| bezier.nearest(*point, NEAREST_PREC).1)
                            .sum();

        return (total_error / n_points as f64, bezier);
    };

    let matrix_cache = MatrixCache::new(points, constraints, dof, M);
    let mut proposal = initial_guess::<T>(points);

    // initialize with placeholders
    // `error` and `new_error` report the _mean_ distance to each of the points
    let mut error = std::f64::INFINITY;
    let mut ts = vec![0.; n_points];
    let mut iteration = 0;

    loop {
        let mut new_error = 0.;

        // find projections
        for i in 0..n_points {
            let point = points[i];
            let (nearest_t, distance) = proposal.nearest(point, NEAREST_PREC);
            new_error += distance;
            ts[i] = nearest_t;
        };

        new_error /= n_points as f64;

        // check if the curve is good enough
        if (new_error - error).abs() < STOP_TOL {
            error = new_error;
            break;
        } else if iteration == MAX_ITER {
            eprintln!("WARNING: max iterations reached when fitting a curve");
            break;
        } else {
            error = new_error;
        };

        // fit a new curve with current projections
        proposal = fit_with_t::<T>(&points, &ts, constraints, dof, M, &matrix_cache);

        iteration += 1;
    };

    (error, proposal)
}

#[cfg(test)]
mod test {
    use crate::{
        Point,
        fitting::{
            Constraint, DMatrix, build_m, build_embedding, two_block, VectorN
        },
        cubicfit::M_8
    };
    use Constraint::*;

    #[test]
    fn test_two_block() {
        let block = DMatrix::from_row_slice(3, 2, &[
            1., 2.,
            3., 4.,
            5., 6.,
        ]);

        let expected = DMatrix::from_row_slice(6, 4, &[
            1., 2., 0., 0., 
            3., 4., 0., 0., 
            5., 6., 0., 0.,
            0., 0., 1., 2.,
            0., 0., 3., 4.,
            0., 0., 5., 6.,

        ]);
        
        let result = two_block(&block);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_build_m_free() {
        let constraints = [
            Fixed((0., 0.).into()),
            Line(Point::new(0., 0.), Point::new(1., 1.)),
            Free,
            Fixed((0., 1.).into()),
        ];

        let m_free = build_m(&constraints, false, 4, &M_8);

        let m_free_expected = DMatrix::from_row_slice(8, 4, &[
             0.,  0.,     0.,  0.,
             3.,  0.,     0.,  0.,
            -6.,  3.,     0.,  0.,
             3., -3.,     0.,  0.,

             0.,  0.,     0.,  0.,
             0.,  0.,     3.,  0.,
             0.,  0.,    -6.,  3.,
             0.,  0.,     3., -3.,
        ]);

        assert_eq!(m_free, m_free_expected);
    }

    #[test]
    fn test_build_m_fixed() {
        let constraints = [
            Fixed((0., 0.).into()),
            Line(Point::new(0., 0.), Point::new(1., 1.)),
            Free,
            Fixed((0., 1.).into()),
        ];

        let m_fixed = build_m(&constraints, true, 4, &M_8);

        let m_fixed_expected = DMatrix::from_row_slice(8, 4, &[
             1.,  0.,    0.,  0.,
            -3.,  0.,    0.,  0.,
             3.,  0.,    0.,  0.,
            -1.,  1.,    0.,  0.,

             0.,  0.,    1.,  0.,
             0.,  0.,   -3.,  0.,
             0.,  0.,    3.,  0.,
             0.,  0.,   -1.,  1.,
        ]);

        assert_eq!(m_fixed, m_fixed_expected);
    }

    #[test]
    fn test_build_embedding() {
        let constraints = [
            Fixed((0., 0.).into()),
            Line(Point::new(2., 0.), Point::new(1., 0.)),
            Free,
            Fixed((0., 1.).into()),
        ];

        let (embedding, additive) = build_embedding(&constraints);

        let embedding_expected = DMatrix::from_row_slice(4, 3, &[
            1., 0., 0.,
            0., 1., 0.,
            0., 0., 0.,
            0., 0., 1.,
        ]);

        let additive_expected = VectorN::from_row_slice(&[
            1.,
            0.,
            0.,
            0.
        ]);

        assert_eq!(embedding, embedding_expected);
        assert_eq!(additive, additive_expected);
    }
}
