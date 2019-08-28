use crate::{CubicBez, Point, Line, ParamCurveFit};

use nalgebra::{
    U1, U2, U4, Dynamic, Matrix, VecStorage,
    Vector2 as GVector2, Matrix4, linalg::LU
};
use lazy_static::lazy_static;

type VMatrix<Col, Row> = Matrix<f64, Col, Row, VecStorage<f64, Col, Row>>;

type MatrixNx4 = VMatrix<Dynamic, U4>;
type Matrix4xN = VMatrix<U4, Dynamic>;
type VectorN = VMatrix<Dynamic, U1>;
type MatrixNx2 = VMatrix<Dynamic, U2>;
type Vector2 = GVector2<f64>;

lazy_static! {
    static ref M_LU_DECOMP: LU<f64, U4, U4> = {
        #[allow(non_snake_case)]
        let M = Matrix4::new(
             1.,  0.,  0.,  0.,
            -3.,  3.,  0.,  0.,
             3., -6.,  3.,  0.,
            -1.,  3., -3.,  1.
        );
        let lu = M.lu();

        lu
    };
}

impl ParamCurveFit for CubicBez {
    #[allow(non_snake_case)]
    fn fit_with_t(points: &[Point], ts: &[f64]) -> Self {
        let T = MatrixNx4::from_fn(ts.len(), |r, c| ts[r].powi(c as i32));
        let T_T = T.transpose();
        let T_TT_lu = (&T_T * T).lu();

        fn fit_single_coord(
            n_points: usize,
            coord: impl Iterator<Item=f64>,
            t_tt_lu: &LU<f64, U4, U4>,
            t_t: &Matrix4xN,
            ) -> [f64; 4]
        {
            let P = VectorN::from_iterator(n_points, coord);

            let T_TP = t_t * P;

            let solve_1 = t_tt_lu.solve(&T_TP).expect("solve_1 failed");
            let solve_2 = M_LU_DECOMP.solve(&solve_1).expect("solve_2 failed");

            [solve_2[0], solve_2[1], solve_2[2], solve_2[3]]
        };

        let xs = fit_single_coord(
            points.len(),
            points.iter().map(|p| p.x),
            &T_TT_lu,
            &T_T
        );
        let ys = fit_single_coord(
            points.len(),
            points.iter().map(|p| p.y),
            &T_TT_lu,
            &T_T
        );

        CubicBez::new(
            Point::new(xs[0], ys[0]),
            Point::new(xs[1], ys[1]),
            Point::new(xs[2], ys[2]),
            Point::new(xs[3], ys[3])
        )
    }


    fn initial_guess(points: &[Point]) -> Self {
        let p0 = points.first().expect("failed to fetch the first point");
        let pn = points.last().expect("failed to fetch the last point");

        Line::new(*p0, *pn).into()
    }
}

/// Represents the three possible types of constraints imposed on control points
/// when fitting curves
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Constraint {
    /// The control point is unconstrained
    Free,
    /// The control point is fixed to be exactly equal to `Point`
    Fixed(Point),
    /// The control point is constrained to lie on the line
    /// between the two points
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

    fn is_line(&self) -> bool {
        if let Constraint::Line(_, _) = *self {
            true
        } else {
            false
        }
    }
}

fn initial_guess(points: &[Point]) -> CubicBez {
    let p0 = points.first().expect("failed to fetch the first point");
    let pn = points.last().expect("failed to fetch the last point");

    Line::new(*p0, *pn).into()
}

type DMatrix = VMatrix<Dynamic, Dynamic>;

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

lazy_static! {
    static ref M_8: DMatrix = {
        // This matrix is actually block-diagonal and could be simplified as
        //
        // M_8 = [[M, 0],
        //        [0, M]]
        //
        // where M is the matrix defined in
        // https://pomax.github.io/bezierinfo/#curvefitting
        DMatrix::from_row_slice(8, 8, &[
             1.,  0.,  0.,  0.,    0.,  0.,  0.,  0.,
            -3.,  3.,  0.,  0.,    0.,  0.,  0.,  0.,
             3., -6.,  3.,  0.,    0.,  0.,  0.,  0.,
            -1.,  3., -3.,  1.,    0.,  0.,  0.,  0.,

             0.,  0.,  0.,  0.,    1.,  0.,  0.,  0.,
             0.,  0.,  0.,  0.,   -3.,  3.,  0.,  0.,
             0.,  0.,  0.,  0.,    3., -6.,  3.,  0.,
             0.,  0.,  0.,  0.,   -1.,  3., -3.,  1.,
        ])
    };
}

fn build_m(constraints: &[Constraint; 4], fixed: bool) -> DMatrix {
    let mut columns = Vec::<usize>::with_capacity(8);

    // pick X coordinate columns (0..4)
    for (i, constr) in constraints.iter().enumerate() {
        if fixed == constr.is_fixed() {
            columns.push(i);
        }
    }

    // pick Y coordinate columns (4..8)
    for (i, constr) in constraints.iter().enumerate() {
        if fixed == constr.is_fixed() {
            columns.push(i + 4);
        }
    }
    
    M_8.select_columns(columns.iter())
}

fn embed_add(a: &Point, b: &Point) -> (Vector2, Vector2) {
    let embed = Vector2::new(a.x - b.x, a.y - b.y);
    let add = Vector2::new(b.x, b.y);

    (embed, add)
}

fn build_embedding(constraints: &[Constraint; 4]) -> (DMatrix, VectorN) {
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

fn build_mc_offset(constraints: &[Constraint; 4]) -> VectorN {
    let m = build_m(&constraints, true);
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

#[allow(non_snake_case)]
fn fit_with_t(points: &[Point], ts: &[f64], constraints: &[Constraint; 4]) -> CubicBez {
    assert_eq!(points.len(), ts.len());
    let n_points = points.len();

    let M_free = build_m(constraints, false);

    // T is of shape (2 * n_points, 2 * 4) in order to map from the
    // 8 coordinates of control points (4 x,y pairs) to the 2 * n_points
    // coordinates of each fitted point
    let T = {
        let T_small = DMatrix::from_fn(n_points, 4, |r, c| {
            ts[r].powi(c as i32)
        });

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
        let mc_offset = build_mc_offset(constraints);
        &T * mc_offset
    };

    let (embedding, additive) = build_embedding(constraints);

    let T_M_free = T * M_free;
    let T_M_free_embedded = &T_M_free * embedding;
    let additive_offset = T_M_free * additive;
    let P = P_base - P_offset - additive_offset;

    // now we have T, M and P and can solve least squares for C
    let svd = T_M_free_embedded.svd(true, true);
    let least_sqr = svd.solve(&P, 0.).expect("solve failed");

    // and reconstruct the bezier control points
    let mut ctrl_points = Vec::with_capacity(4);
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

    CubicBez {
        p0: ctrl_points[0],
        p1: ctrl_points[1],
        p2: ctrl_points[2],
        p3: ctrl_points[3]
    }
}

fn fit(points: &[Point], constraints: &[Constraint; 4]) -> (f64, CubicBez) {
    use crate::ParamCurveNearest;

    const NEAREST_PREC: f64 = 1e-6; // TODO: how much?
    const STOP_TOL: f64 = 1.; // TODO: how much?

    let n_points = points.len();

    let mut proposal = initial_guess(points);

    // initialize with placeholders
    // `error` and `new_error` report the _mean_ distance to each of the points
    let mut error = std::f64::INFINITY;
    let mut ts = vec![0.; n_points];

    loop {
        let mut new_error = 0.;

        // find projections
        for i in 0..n_points {
            let point = points[i];
            let (nearest_t, distance) = proposal.nearest(point, NEAREST_PREC);
            new_error += distance;
            ts[i] = nearest_t;
        };

        new_error /= (n_points as f64);

        dbg!("new iteration", error, new_error);
        // check if the curve is good enough
        if (new_error - error).abs() < STOP_TOL {
            error = new_error;
            break;
        } else {
            error = new_error;
        };

        // fit a new curve with current projections
        proposal = fit_with_t(&points, &ts, constraints);
    };

    (error, proposal)
}

mod test {
    use crate::{
        Point, CubicBez, ParamCurveFit, AbsDiffEq, assert_abs_diff_eq,
        fitting::{
            M_8, Constraint, DMatrix, build_m, build_embedding, fit, two_block, VectorN
        }
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

        let m_free = build_m(&constraints, false);

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

        let m_fixed = build_m(&constraints, true);

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

    #[test]
    fn test_fitting_regression() {
        let points = [
            Point::new(0., 0.),
            Point::new(90., 60.),
            Point::new(30., 10.),
            Point::new(50., 30.),
            Point::new(60., 20.),
            Point::new(80., 15.),
            Point::new(65., 40.)
        ];

        let constraints: [Constraint; 4] = [
            Free,
            Free,
            Free,
            Free
        ];

        let (error, new_cubic_bez) = fit(&points, &constraints);
        let old_cubic_bez = CubicBez::fit(&points);

        assert_abs_diff_eq!(new_cubic_bez, old_cubic_bez);
    }

    #[test]
    fn test_fitting_fixed() {
        let points = [
            Point::new(0., 0.),
            Point::new(90., 60.),
            Point::new(30., 10.),
            Point::new(50., 30.),
            Point::new(60., 20.),
            Point::new(80., 15.),
            Point::new(65., 40.)
        ];

        let constraints: [Constraint; 4] = [
            Fixed((5., 5.).into()),
            Free,
            Free,
            Free
        ];
        
        let (error, curve) = fit(&points, &constraints);
    }

    #[test]
    fn test_fitting_line() {
        let points = [
            Point::new(0., 0.),
            Point::new(90., 60.),
            Point::new(30., 10.),
            Point::new(50., 30.),
            Point::new(60., 20.),
            Point::new(80., 15.),
            Point::new(65., 40.)
        ];

        let constraints: [Constraint; 4] = [
            Line(Point::new(0., 0.), Point::new(1., 0.)),
            Free,
            Free,
            Line(Point::new(0., 0.), Point::new(1., 1.)),
        ];
        
        let (error, curve) = fit(&points, &constraints);

        assert_abs_diff_eq!(curve.p0.y, 0.);

        let p3 = curve.p3;
        assert_abs_diff_eq!(p3.x - p3.y, 0.);
    }
}
