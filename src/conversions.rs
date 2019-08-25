use crate::{Line, CubicBez, QuadBez};

impl From<Line> for CubicBez {
    fn from(line: Line) -> CubicBez {
        let p0 = line.p0.to_vec2();
        let p3 = line.p1.to_vec2();

        let control = p0.lerp(p3, 0.5);

        CubicBez::new(
            p0.to_point(),
            control.to_point(),
            control.to_point(),
            p3.to_point()
        )
    }
}

impl From<Line> for QuadBez {
    fn from(line: Line) -> QuadBez {
        let p0 = line.p0.to_vec2();
        let p2 = line.p1.to_vec2();

        let control = p0.lerp(p2, 0.5);

        QuadBez::new(
            p0.to_point(),
            control.to_point(),
            p2.to_point()
        )
    }
}

impl From<QuadBez> for CubicBez {
    fn from(quad: QuadBez) -> CubicBez {
        CubicBez::new(
            quad.p0,
            quad.p1,
            quad.p1,
            quad.p2
        )
    }
}
