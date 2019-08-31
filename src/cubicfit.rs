use crate::{
    Constraint, CubicBez, Point, Line, ParamCurveFit,
    fitting::{DMatrix, fit, FromPointIter}
};
use lazy_static::lazy_static;

lazy_static! {
    pub(crate) static ref M_8: DMatrix = {
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

impl FromPointIter for CubicBez {
    fn from_point_iter(mut iter: impl Iterator<Item = Point>) -> Self {
        let p0 = iter.next().unwrap();
        let p1 = iter.next().unwrap();
        let p2 = iter.next().unwrap();
        let p3 = iter.next().unwrap();
        assert!(iter.next().is_none(), "iterator not exhausted");

        CubicBez { p0, p1, p2, p3 }
    }
}

impl ParamCurveFit for CubicBez {
    type Constraints = [Constraint; 4];

    fn fit(points: &[Point], constraints: &Self::Constraints) -> (f64, Self) {
        fit::<CubicBez>(points, constraints, 4, &M_8)
    }
}

#[cfg(test)]
mod test {
    use crate::{
        Constraint, Point, CubicBez, assert_abs_diff_eq, ParamCurveFit,
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

        let constraints: [Constraint; 4] = [
            Fixed((5., 5.).into()),
            Free,
            Free,
            Free
        ];
        
        let (_error, curve) = CubicBez::fit(&points, &constraints);
        assert_eq!(curve.p0, Point::new(5., 5.));
    }

    #[test]
    fn test_fitting_4() {
        let points = [
            Point::new(5., 7.),
            Point::new(3., 2.),
            Point::new(90., 60.),
            Point::new(30., 10.),
        ];

        let constraints = [Free, Free, Free, Free];

        let (error, _curve) = CubicBez::fit(&points, &constraints);
        assert!(error < 5.);
    }

    #[test]
    fn test_fitting_3() {
        let points = [
            Point::new(5., 7.),
            Point::new(3., 2.),
            Point::new(30., 10.),
        ];

        let constraints = [Free, Free, Free, Free];

        let (error, _curve) = CubicBez::fit(&points, &constraints);
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
            Point::new(0., 2.),
            Point::new(3., 3.),

        ];

        let constraints: [Constraint; 4] = [
            ctrl_points[0].into(),
            ctrl_points[1].into(),
            ctrl_points[2].into(),
            ctrl_points[3].into(),
        ];

        let (_error, curve) = CubicBez::fit(&points, &constraints);
        let expected = CubicBez::new(
            ctrl_points[0],
            ctrl_points[1],
            ctrl_points[2],
            ctrl_points[3],
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

        let constraints: [Constraint; 4] = [
            Line(Point::new(0., 0.), Point::new(1., 0.)),
            Free,
            Free,
            Line(Point::new(0., 0.), Point::new(1., 1.)),
        ];
        
        let (_error, curve) = CubicBez::fit(&points, &constraints);

        assert_abs_diff_eq!(curve.p0.y, 0.);

        let p3 = curve.p3;
        assert_abs_diff_eq!(p3.x - p3.y, 0.);
    }
}
