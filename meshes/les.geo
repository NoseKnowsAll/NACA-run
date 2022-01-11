// Wing
Include "naca0012.geo";
Spline(1) = { 1:53 };
Spline(2) = { 53:104,1 };
Line Loop(1) = { 1,2 };

// Outer boundary rectangle
R = 10;
Point(411) = { .5-R,-R,0 };
Point(412) = { .5+2*R,-R,0 };
Point(413) = { .5+2*R, R,0 };
Point(414) = { .5-R, R,0 };

Line(5) = { 411, 412 };
Line(6) = { 412, 413 };
Line(7) = { 413, 414 };
Line(8) = { 414, 411 };

Line Loop(2) = { 5, 6, 7, 8 };

// Final geometry
Plane Surface(1) = { 2,1 };

Physical Line("Airfoil", 1) = {1,2};
Physical Line("Far field", 2) = {5,6,7,8};
Physical Surface("Domain", 1) = {1};

// Size field

hLE = 0.00205;
hTE = 0.002195;
hwing = 0.02;
delta_wing = 0.2;
hwake = 0.02;
hwake2 = 0.1;
delta_wake = 0.15;
delta_wake2 = 0.5;

hshock = 0.015;
delta_shock = 0.2;

h_shk_bl = 0.03;

hgrw = 3;
dgrw = 10;

Field[1] = Attractor;
Field[1].NNodesByEdge = 100;
Field[1].EdgesList = { 1 };

Field[2] = Threshold;
Field[2].IField = 1;
Field[2].LcMin = hwing;
Field[2].LcMax = hwing + hgrw;
Field[2].DistMin = delta_wing;
Field[2].DistMax = delta_wing + dgrw;
Field[2].StopAtDistMax = 1;

Field[3] = Attractor;
Field[3].NodesList = { 53 };

Field[4] = Threshold;
Field[4].IField = 3;
Field[4].LcMin = hLE;
Field[4].LcMax = hLE + hgrw;
Field[4].DistMin = 0;
Field[4].DistMax = dgrw;
Field[4].StopAtDistMax = 1;

Field[5] = Attractor;
Field[5].NNodesByEdge = 100;
Field[5].EdgesList = { 1 };

Field[6] = Threshold;
Field[6].IField = 5;
Field[6].LcMin = hTE;
Field[6].LcMax = hTE + hgrw;
Field[6].DistMin = 0;
Field[6].DistMax = dgrw;
Field[6].StopAtDistMax = 1;

Point(421) = { 0.2,0.3,0 };
Point(422) = { 1.8,0.3,0 };
Line(423) = { 421, 422 };
Field[7] = Attractor;
Field[7].NNodesByEdge = 100;
Field[7].EdgesList = { 423 };

Field[8] = Threshold;
Field[8].IField = 7;
Field[8].LcMin = hwake;
Field[8].LcMax = hwake + hgrw;
Field[8].DistMin = 0.3;
Field[8].DistMax = 0.3 + dgrw;
Field[8].StopAtDistMax = 1;

Point(424) = { 0.5,1,0 };
Point(425) = { 2.5,1,0 };
Line(426) = { 424, 425 };
Field[9] = Attractor;
Field[9].NNodesByEdge = 100;
Field[9].EdgesList = { 426 };

hwake2 = hwake * 2;
Field[10] = Threshold;
Field[10].IField = 9;
Field[10].LcMin = hwake2;
Field[10].LcMax = hwake2 + hgrw;
Field[10].DistMin = 0.4;
Field[10].DistMax = 0.4 + dgrw;
Field[10].StopAtDistMax = 1;

Field[15] = Attractor;
Field[15].NodesList = { 1 };

Field[16] = Threshold;
Field[16].IField = 15;
Field[16].LcMin = 2;
Field[16].LcMax = 2;
Field[16].DistMin = 0;
Field[16].DistMax = 110;
Field[16].StopAtDistMax = 1;

Field[17] = Min;
Field[17].FieldsList = { 2,4,6,8,10,16 };
Background Field = 17;

Mesh.Algorithm = 8;
Mesh.HighOrderOptimize = 1;
Recombine Surface(1);
Mesh.CharacteristicLengthExtendFromBoundary = 0;

Mesh 2;
