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

heddies1 = 0.05;
heddies2 = 0.5;
radius1_eddies = 0.25;
radius2_eddies = 0.5;

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

Point(115) = { 0.25,0.25,0 };
Point(116) = { 0.75,0.25,0 };
Point(117) = { 0.95,1.10,0 };
Point(118) = { 0.25,0.75,0 };
Line(119) = { 115,116 };
Line(120) = { 116,117 };
Line(121) = { 117,118 };
Line(122) = { 118,115 };
Curve Loop(123) = { 119, 120, 121, 122 };

Field[7] = Distance;
Field[7].NNodesByEdge = 100;
Field[7].EdgesList = { 119,120,121,122 };

Field[8] = Threshold;
Field[8].IField = 7;
Field[8].LcMin = heddies1;
Field[8].LcMax = heddies2;
Field[8].DistMin = radius1_eddies;
Field[8].DistMax = radius2_eddies;
Field[8].StopAtDistMax = 1;

Field[11] = Min;
Field[11].FieldsList = { 2,4,6,8 };
Background Field = 11;

Mesh.Algorithm = 8;
Mesh.HighOrderOptimize = 1;
Recombine Surface(1);
Mesh.CharacteristicLengthExtendFromBoundary = 0;
