#![allow(non_snake_case)]
use crate::{CubicBez, Point, Line, ParamCurveNearest, ParamCurveFit};

use nalgebra::{
    U1, Dynamic, Matrix, VecStorage, Vector2 as GVector2
};

type VMatrix<Col, Row> = Matrix<f64, Col, Row, VecStorage<f64, Col, Row>>;

pub(crate) type DMatrix = VMatrix<Dynamic, Dynamic>;
pub(crate) type VectorN = VMatrix<Dynamic, U1>;
type Vector2 = GVector2<f64>;

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
    /// # use kurbo::{Point, Constraint};
    /// let p = Point::new(0., 0.);
    /// assert_eq!(Constraint::Fixed(p), p.into());
    /// ```
    Fixed(Point),
    
    /// This control point is constrained to lie on the line
    /// between the two points.
    /// A `From<(Point, Point)>` implementation is provided
    /// to allow easy construction
    /// ```
    /// # use kurbo::{Point, Constraint};
    /// let p1 = Point::new(0., 0.);
    /// let p2 = Point::new(1., 0.);
    /// assert_eq!(Constraint::Line(p1, p2), (p1, p2).into());
    /// ```
    Line(Point, Point),
}

impl From<Point> for Constraint {
    /// Constructs a `Constraint::Fixed` variant from the point
    /// ```
    /// # use kurbo::{Point, Constraint};
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
    /// # use kurbo::{Point, Constraint};
    /// let p1 = Point::new(0., 0.);
    /// let p2 = Point::new(1., 0.);
    /// assert_eq!(Constraint::Line(p1, p2), (p1, p2).into());
    /// ```
    fn from(ps: (Point, Point)) -> Constraint {
        Constraint::Line(ps.0, ps.1)
    }
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

// TODO: improve performance by caching matrices such as `P_base`, `embedding`, `additive`,
// `P_offset` inside the `fit` parent instead of recomputing them each time inside `fit_with_t`
fn fit_with_t<T: FromPointIter>(
    points: &[Point], ts: &[f64], constraints: &[Constraint],
    dof: usize, M: &DMatrix) -> T
{
    assert_eq!(constraints.len(), dof);
    assert_eq!(M.shape().0, 2 * dof);
    assert_eq!(M.shape().1, 2 * dof);
    assert_eq!(points.len(), ts.len());
    let n_points = points.len();

    let M_free = build_m(constraints, false, dof, &M);

    // T is of shape (2 * `n_points`, 2 * `dof`) in order to map from the
    // 2 * `dof` coordinates of control points (`dof` x,y pairs) to the 2 * `n_points`
    // coordinates of each fitted point
    let T = {
        let T_small = DMatrix::from_fn(n_points, dof, |r, c| {
            ts[r].powi(c as i32)
        });

        dbg!(ts, &T_small);
        two_block(&T_small)
    };

    let P_base = {
        let xs = points
                    .iter()
                    .map(|p| p.x);
        let ys = points
                    .iter()
                    .map(|p| p.y);

        VectorN::from_iterator(points.len() * 2, xs.chain(ys))
    };

    let P_offset = {
        let mc_offset = build_mc_offset(constraints, dof, M);
        &T * mc_offset
    };

    let (embedding, additive) = build_embedding(constraints);

    dbg!(&P_base, &P_offset, &T, &M_free);
    let T_M_free = &T * &M_free; // FIXME: T can be taken by value
    let T_M_free_embedded = &T_M_free * embedding;
    let additive_offset = T_M_free * additive;
    let P = P_base - P_offset - additive_offset;

    // now we have T, M and P and can solve least squares for C
    let svd = T_M_free_embedded.svd(true, true);
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


    let mut proposal = initial_guess::<T>(points);
    dbg!(&proposal);

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
            dbg!(&point, &nearest_t, &distance);
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
        proposal = fit_with_t::<T>(&points, &ts, constraints, dof, M);

        iteration += 1;
    };

    (error, proposal)
}

#[cfg(test)]
mod test {
    use crate::{
        Point, CubicBez, assert_abs_diff_eq, ParamCurveFit,
        fitting::{
            Constraint, DMatrix, build_m, build_embedding, fit, two_block, VectorN
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
