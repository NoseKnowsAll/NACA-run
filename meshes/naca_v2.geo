// NACA mesh

// Wing
Include "naca0012.geo";
Spline(1) = { 1:53 };
Spline(2) = { 53:104,1 };
Line Loop(1) = { 1,2 };

// Outer boundary rectangle
R = 5;
Point(111) = { .5-R,-R,0 };
Point(112) = { .5+2*R,-R,0 };
Point(113) = { .5+2*R, R,0 };
Point(114) = { .5-R, R,0 };

Line(5) = { 111, 112 };
Line(6) = { 112, 113 };
Line(7) = { 113, 114 };
Line(8) = { 114, 111 };

Line Loop(2) = { 5, 6, 7, 8 };

// Final geometry
Plane Surface(1) = { 2,1 };

Physical Line("Airfoil", 1) = {1,2};
Physical Line("Far field", 2) = {5,6,7,8};
Physical Surface("Domain", 1) = {1};

// Size field

// These two values are set by external scripts such as find_hs.m with parameter "-setnumber hLE XXX" "-setnumber hwing YYY"
//hLE = 0.01;
hTE = 0.01;
//hwing = 0.05;
delta_wing = 0.2;

hgrw = 3;
dgrw = 10;

Field[1] = Distance;
Field[1].NNodesByEdge = 100;
Field[1].EdgesList = { 1,2 };

Field[2] = Threshold;
Field[2].IField = 1;
Field[2].LcMin = hwing;
Field[2].LcMax = hwing + hgrw;
Field[2].DistMin = delta_wing;
Field[2].DistMax = delta_wing + dgrw;
Field[2].StopAtDistMax = 1;

Field[3] = Distance;
Field[3].NodesList = { 53 };

Field[4] = Threshold;
Field[4].IField = 3;
Field[4].LcMin = hLE;
Field[4].LcMax = hLE + hgrw;
Field[4].DistMin = 0;
Field[4].DistMax = dgrw;
Field[4].StopAtDistMax = 1;

Field[5] = Distance;
Field[5].NodesList = { 1 };

Field[6] = Threshold;
Field[6].IField = 5;
Field[6].LcMin = hTE;
Field[6].LcMax = hTE + hgrw;
Field[6].DistMin = 0;
Field[6].DistMax = dgrw;
Field[6].StopAtDistMax = 1;

Field[11] = Min;
Field[11].FieldsList = { 2,4,6 };
Background Field = 11;

Mesh.Algorithm = 8;
Mesh.HighOrderOptimize = 1;
Recombine Surface(1);
Mesh.CharacteristicLengthExtendFromBoundary = 0;
