use crate::{CubicBez, Point, Line, ParamCurveFit};

use nalgebra::{U1, U2, U4, Dynamic, Matrix, VecStorage, Matrix4, linalg::LU};
use lazy_static::lazy_static;

type VMatrix<Col, Row> = Matrix<f64, Col, Row, VecStorage<f64, Col, Row>>;

type MatrixNx4 = VMatrix<Dynamic, U4>;
type Matrix4xN = VMatrix<U4, Dynamic>;
type VectorN = VMatrix<Dynamic, U1>;
type MatrixNx2 = VMatrix<Dynamic, U2>;

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
        fn initial_guess_inner(points: &[Point]) -> CubicBez {
            let p0 = points.first().expect("failed to fetch the first point");
            let pn = points.last().expect("failed to fetch the last point");

            Line::new(*p0, *pn).into()
        }

        //let matrix = MatrixNx2::from_fn(points.len(), |r, c| {
        //    let point = points[r];

        //    if c == 0 {
        //        point.x
        //    } else {
        //        point.y
        //    }
        //});

        //let svd = matrix.clone().svd(true, true);
        //let u = svd.u.unwrap();
        //let mut s = svd.singular_values;
        //let v_t = svd.v_t.unwrap();

        //let m = u * s * v_t;

        //dbg!(&matrix, &m);
        //dbg!(&u, &s, &v_t);

        initial_guess_inner(points)
    }
}
