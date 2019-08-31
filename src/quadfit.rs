use crate::{
    Point, QuadBez, ParamCurveFit, Constraint,
    fitting::{DMatrix, VectorN}
};
use lazy_static::lazy_static;

impl ParamCurveFit for QuadBez {
    type Constraints = [Constraint; 3];

    fn fit(points: &[Point], constraints: &Self::Constraints) -> (f64, Self) {
        unimplemented!();
    }
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

#[cfg(test)]
mod test {
    use crate::{
        Point, QuadBez, assert_abs_diff_eq, ParamCurveFit,
        fitting::{
            Constraint, DMatrix, build_m, build_embedding, fit, two_block, VectorN
        }
    };
    use Constraint::*;

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

        let constraints: [Constraint; 3] = [
            Fixed((5., 5.).into()),
            Free,
            Free,
        ];
        
        let (_error, curve) = QuadBez::fit(&points, &constraints);
        assert_eq!(curve.p0, Point::new(5., 5.));
    }

    #[test]
    fn test_fitting_3() {
        let points = [
            Point::new(5., 7.),
            Point::new(3., 2.),
            Point::new(90., 60.),
        ];

        let constraints = [Free, Free, Free];

        let (error, _curve) = QuadBez::fit(&points, &constraints);
        assert!(error < 5.);
    }

    #[test]
    fn test_fitting_2() {
        let points = [
            Point::new(5., 7.),
            Point::new(3., 2.),
        ];

        let constraints = [Free, Free, Free];

        let (error, _curve) = QuadBez::fit(&points, &constraints);
        assert!(error < 5.);
    }

    #[test]
    fn test_fitting_no_dof() {
        let points = [
            Point::new(0., 0.),
            Point::new(90., 60.),
            Point::new(30., 10.),
            Point::new(50., 30.),
            Point::new(60., 20.),
            Point::new(80., 15.),
            Point::new(65., 40.)
        ];

        let ctrl_points = [
            Point::new(0., 0.),
            Point::new(1., 0.),
            Point::new(3., 3.),

        ];

        let constraints: [Constraint; 3] = [
            ctrl_points[0].into(),
            ctrl_points[1].into(),
            ctrl_points[2].into(),
        ];

        let (_error, curve) = QuadBez::fit(&points, &constraints);
        let expected = QuadBez::new(
            ctrl_points[0],
            ctrl_points[1],
            ctrl_points[2],
        );

        assert_abs_diff_eq!(curve, expected);
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

        let constraints: [Constraint; 3] = [
            Line(Point::new(0., 0.), Point::new(1., 0.)),
            Free,
            Line(Point::new(0., 0.), Point::new(1., 1.)),
        ];
        
        let (_error, curve) = QuadBez::fit(&points, &constraints);

        assert_abs_diff_eq!(curve.p0.y, 0.);

        let p2 = curve.p2;
        assert_abs_diff_eq!(p2.x - p2.y, 0.);
    }
}
